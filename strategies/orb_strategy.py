"""
Opening Range Breakout (ORB) strategy, used when regime_detector classifies
the session as TRENDING.
"""

import asyncio
import datetime
import enum
import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from google.genai import types

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrderByIdRequest

from utils.alpaca_client import trade_client, data_client
from strategies.gemini_reasoning import ai_client
from utils.telegram_notifier import TelegramNotifier
from utils.trade_logger import log_trade_to_journal
from utils.split_order_tracker import register_single_order
from config.settings import RISK_DOLLARS_PER_TRADE, MAX_NOTIONAL_PER_TRADE

# --- Tunable thresholds --------------------------------------------------
ORB_WINDOW_MINUTES = 15            # 15-minute opening range
MIN_OR_WIDTH_ATR = 0.15            # skip if range is too narrow relative to ATR
MAX_CHASE_ATR = 1.0                # skip if price is past this ATR boundary cap

TP_SL_RATIO = 1.5                  # TP distance = SL distance x 1.5

LIQUIDITY_PROXIMITY_ATR = 1.0      # anchor-scale: how close a prior-day level must sit beyond
                                   # the OR boundary to suspect a liquidity-sweep fakeout
VALUE_AREA_PCT = 0.70              # standard market-profile convention
VOLUME_PROFILE_BINS = 15           # price buckets across the opening-range window
FAKEOUT_WATCH_TIMEOUT_MINUTES = 45 # abandon the watch (and skip ORB today) if no confirming
                                   # reversal candle shows up in this window


# --- Operational Time Cutoffs ---
ORB_DEADLINE = datetime.time(10, 15)       # Stop taking new fresh breakouts after 10:15 AM ET
FAKEOUT_DEADLINE = datetime.time(11, 0)    # Drop lingering fakeout watches entirely after 11:00 AM ET


_orb_fakeout_watch: dict[str, dict] = {}   # per-group_name, in-memory only

class TrendAction(str, enum.Enum):
    BUY = "BUY"
    HOLD = "HOLD"


class ORBTradingSignal(BaseModel):
    mathematical_proof: str = Field(
        description="Step-by-step: opening range bounds, the breakout candle, "
                    "volume/momentum evidence for continuation, and where the "
                    "setup would be invalidated."
    )
    action: TrendAction = Field(description="BUY confirms the breakout as tradeable; HOLD vetoes it as a likely fakeout.")
    confidence_score: int = Field(description="1-100. 85-100=A-Tier clean breakout w/ momentum. 65-84=B-Tier valid but weaker. <65=C-Tier, must be HOLD.")
    reasoning: str = Field(description="Brief explanation of the continuation thesis.")


# --- Up to 2 trades per day ---
_orb_trade_counts: dict[str, tuple[datetime.date, int]] = {}  # group_name -> (date, count)
MAX_ORB_TRADES_PER_DAY = 2

def _extract_bracket_legs(order):
    """
    Extracts take_profit and stop_loss child leg orders 
    from an Alpaca bracket order object.
    """
    if not hasattr(order, "legs") or not order.legs:
        return None, None

    tp_leg = None
    sl_leg = None

    for leg in order.legs:
        stop_price = getattr(leg, "stop_price", None)
        limit_price = getattr(leg, "limit_price", None)
        order_type = str(getattr(leg, "order_type", "")).lower()

        # Identify Stop Loss leg
        if stop_price is not None or "stop" in order_type:
            sl_leg = leg
        # Identify Take Profit leg
        elif limit_price is not None or "limit" in order_type:
            tp_leg = leg

    return tp_leg, sl_leg


async def _submit_orb_trade(group_name, execution_ticker, entry_price, sl_price, tp_price,
                             confidence, reasoning, math_proof, setup_type, anchor, ctx,
                             extra_snapshot=None) -> bool:
    
    # --- SANITY CHECK FOR BUY BRACKET ORDERS ---
    if sl_price >= (entry_price - 0.01):
        corrected_sl = round(entry_price - max(abs(entry_price - sl_price), 0.50), 2)
        print(f"⚠️ [ORB] Inverted SL detected for {execution_ticker}. Corrected {sl_price} -> {corrected_sl}")
        sl_price = corrected_sl

    if tp_price <= (entry_price + 0.01):
        corrected_tp = round(entry_price + max(abs(tp_price - entry_price), 1.00), 2)
        print(f"⚠️ [ORB] Inverted TP detected for {execution_ticker}. Corrected {tp_price} -> {corrected_tp}")
        tp_price = corrected_tp
    # -------------------------------------------

    qty = calculate_shares(entry_price, sl_price, RISK_DOLLARS_PER_TRADE)
    if qty == 0:
        print(f"⚠️ [ORB] Invalid stop distance for {execution_ticker}. Skipping.")
        return False
    tier_str = f"{'A' if confidence >= 85 else 'B'}-Tier ORB ({qty} Share{'s' if qty != 1 else ''})"

    buy_request = MarketOrderRequest(
        symbol=execution_ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=tp_price),
        stop_loss=StopLossRequest(stop_price=sl_price),
    )
    
    # 1. Submit market bracket order to Alpaca
    try:
        order = await asyncio.to_thread(trade_client.submit_order, order_data=buy_request)
    except Exception as e:
        print(f"❌ [ORB] Order submission failed for {execution_ticker}: {e}")
        return False

    # 2. Extract child legs, polling up to 6 times with nested=True
    tp_leg, sl_leg = _extract_bracket_legs(order)

    if not all([tp_leg, sl_leg]):
        for attempt in range(6):
            await asyncio.sleep(0.5)  # Pause for Alpaca matching engine
            try:
                order = await asyncio.to_thread(
                    trade_client.get_order_by_id, 
                    order.id, 
                    nested=True  # Correct keyword argument for alpaca-py
                )
                tp_leg, sl_leg = _extract_bracket_legs(order)
                if all([tp_leg, sl_leg]):
                    break
            except Exception as e:
                print(f"⚠️ [ORB] Error fetching order legs: {e}")

    # 3. Register for tracking if legs resolved
    if all([tp_leg, sl_leg]):
        await register_single_order(str(tp_leg.id), str(sl_leg.id), entry_price, execution_ticker)
    else:
        print(f"⚠️ [ORB] Could not resolve bracket child leg ids for {execution_ticker}. "
              f"Position is live on Alpaca with TP/SL attached, but local tracking is unlinked.")

    # 4. ALWAYS log and send Telegram notification for live entries
    snapshot = {
        "setup_type": setup_type, "confidence_score": confidence,
        "anchor": anchor, "anchor_price": ctx["latest_close"],
        "stop_loss": sl_price, "take_profit": tp_price, "mathematical_proof": math_proof,
    }
    if extra_snapshot:
        snapshot.update(extra_snapshot)

    await log_trade_to_journal(
        ticker=execution_ticker, action="BUY", fill_price=entry_price, quantity=qty,
        reasoning=reasoning, market_snapshot=snapshot,
    )
    
    await TelegramNotifier.send(
        f"🚨 *ORB STRATEGY ORDER EXECUTED* 🚨\n\n"
        f"• *Ticker Asset* : {execution_ticker}\n"
        f"• *Strategy Path*: {group_name} ({setup_type})\n"
        f"• *Conviction*   : {tier_str} | Score: {confidence}\n"
        f"• *Entry Price*  : ${entry_price:.2f}\n"
        f"• *Protective SL*: ${sl_price:.2f}\n"
        f"🎯 *Target (TP)*: ${tp_price:.2f}\n\n"
        f"💡 *AI Rationale*: {reasoning}"
    )
    
    return True

def _orb_trade_count_today(group_name: str, today: datetime.date) -> int:
    entry = _orb_trade_counts.get(group_name)
    if entry and entry[0] == today:
        return entry[1]
    return 0


def _mark_orb_traded(group_name: str, today: datetime.date) -> None:
    current_count = _orb_trade_count_today(group_name, today)
    _orb_trade_counts[group_name] = (today, current_count + 1)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    series = _true_range(df).rolling(period, min_periods=period).mean()
    val = series.iloc[-1]
    return float(val) if not np.isnan(val) else 0.0


def calculate_shares(entry_price: float, stop_loss: float, risk_dollars: float,
                      max_notional: float = MAX_NOTIONAL_PER_TRADE) -> int:
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0:
        return 0
    risk_based_qty = max(1, int(risk_dollars / risk_per_share))
    notional_capped_qty = max(1, int(max_notional / entry_price))
    return min(risk_based_qty, notional_capped_qty)


def compute_value_area(bars_df: pd.DataFrame, num_bins: int = VOLUME_PROFILE_BINS,
                        value_area_pct: float = VALUE_AREA_PCT):
    if bars_df.empty or "volume" not in bars_df.columns:
        return None

    price_low = float(bars_df["low"].min())
    price_high = float(bars_df["high"].max())
    if price_high <= price_low:
        return None

    bin_edges = np.linspace(price_low, price_high, num_bins + 1)
    bin_volumes = np.zeros(num_bins)

    typical_prices = (bars_df["high"] + bars_df["low"] + bars_df["close"]) / 3.0
    bin_indices = np.clip(np.digitize(typical_prices, bin_edges) - 1, 0, num_bins - 1)
    for idx, vol in zip(bin_indices, bars_df["volume"]):
        bin_volumes[idx] += float(vol)

    total_volume = bin_volumes.sum()
    if total_volume <= 0:
        return None

    poc_idx = int(np.argmax(bin_volumes))
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    included_low, included_high = poc_idx, poc_idx
    cumulative = bin_volumes[poc_idx]
    while cumulative < total_volume * value_area_pct:
        vol_above = bin_volumes[included_high + 1] if included_high + 1 < num_bins else -1
        vol_below = bin_volumes[included_low - 1] if included_low - 1 >= 0 else -1
        if vol_above < 0 and vol_below < 0:
            break
        if vol_above >= vol_below:
            included_high += 1
            cumulative += bin_volumes[included_high]
        else:
            included_low -= 1
            cumulative += bin_volumes[included_low]

    return round(bin_edges[included_low], 2), round(bin_edges[included_high + 1], 2), round(poc_price, 2)


async def _fetch_minute_bars(symbol: str, start: datetime.datetime, end: datetime.datetime, ny_tz) -> pd.DataFrame:
    resp = await asyncio.to_thread(data_client.get_stock_bars, StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start, end=end, feed=DataFeed.IEX,
    ))
    df = resp.df
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol)
    elif "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize("UTC").tz_convert(ny_tz) if df.index.tz is None else df.index.tz_convert(ny_tz)
    return df


def _detect_liquidity_conflict(direction: str, or_high: float, or_low: float, atr: float, ctx: dict):
    if direction == "BULLISH":
        level = ctx.get("prev_day_high", 0.0)
        if level > or_high and (level - or_high) <= (atr * LIQUIDITY_PROXIMITY_ATR):
            return True, level
    else:
        level = ctx.get("prev_day_low", 0.0)
        if 0 < level < or_low and (or_low - level) <= (atr * LIQUIDITY_PROXIMITY_ATR):
            return True, level
    return False, None


def get_opening_range(df_history: pd.DataFrame, session_start: datetime.datetime, now: datetime.datetime,
                       window_minutes: int = ORB_WINDOW_MINUTES):
    df_history.index = pd.to_datetime(df_history.index, utc=True).tz_convert('America/New_York')
    target_tz = session_start.tzinfo

    if df_history.index.tz is None:
        df_history.index = df_history.index.tz_localize("UTC").tz_convert(target_tz)
    else:
        df_history.index = df_history.index.tz_convert(target_tz)

    range_end = session_start + datetime.timedelta(minutes=window_minutes)
    if now < range_end:
        return None

    or_bars = df_history[(df_history.index >= session_start) & (df_history.index < range_end)]
    if or_bars.empty:
        return None

    or_high = float(or_bars["high"].max())
    or_low = float(or_bars["low"].min())
    or_width = or_high - or_low
    if or_width <= 0:
        return None

    atr = _atr(df_history)
    if atr <= 0:
        return None

    if (or_width / atr) < MIN_OR_WIDTH_ATR:
        return None

    return or_high, or_low, or_width, atr


def detect_orb_breakout(latest_close: float, or_high: float, or_low: float):
    if latest_close > or_high:
        return "BULLISH"
    if latest_close < or_low:
        return "BEARISH"
    return None


def _calculate_orb_levels(entry_price: float, exec_or_low: float):
    """
    Pins the Stop Loss directly to the execution ticker's 15-minute
    Opening Range Low (exec_or_low). Take Profit is derived from 
    TP_SL_RATIO x risk distance.
    """
    sl_price = round(exec_or_low, 2)
    risk_dist = entry_price - sl_price
    if risk_dist <= 0:
        return None, None

    tp_price = round(entry_price + (risk_dist * TP_SL_RATIO), 2)
    return sl_price, tp_price


def _build_orb_prompt(anchor, execution_ticker, direction, multiplier, entry_price,
                       or_high, or_low, atr, ctx) -> str:
    return f"""
You are an expert algorithmic trading engine executing an Opening Range Breakout
(ORB) strategy on a confirmed TRENDING session. A deterministic regime
classifier has already established today looks like a trend day (not a
choppy/ranging one), and a {ORB_WINDOW_MINUTES}-minute opening range has
already been mechanically broken to the {"upside" if direction == "BULLISH" else "downside"}
on a closing basis (not just a wick). Your job is NOT to decide direction, and
NOT to set stop/target prices -- both are already fixed mechanically. Your
only job is to judge whether this looks like genuine trend continuation
rather than an immediate failed breakout, and score your confidence.

--- OPENING RANGE STRUCTURE ({anchor}) ---
* Opening range window     : first {ORB_WINDOW_MINUTES} minutes of the session
* Opening range high       : ${or_high:.2f}
* Opening range low        : ${or_low:.2f}
* 14-period ATR (anchor)   : ${atr:.2f}
* Breakout direction       : {direction}
* Current anchor price     : ${ctx['latest_close']:.2f}
* Current time (ET)        : {ctx['current_time_str']}

--- RECENT 40-CANDLE MATRIX ({anchor}) ---
{ctx['recent_bars_json']}

--- YOUR TASK ---
1. Look at price action since the breakout: are subsequent candles showing
   continuation (higher lows for a bullish break, lower highs for a bearish
   break), or does it look like an immediate rejection back toward the range?
2. Only mark action=BUY if this looks like genuine continuation. If price has
   already round-tripped back toward or inside the opening range, or momentum
   looks exhausted, output HOLD -- this is meant to filter fakeouts, not chase
   every breakout mechanically.
3. Score confidence_score 1-100: 85-100 = clean break, strong displacement,
   momentum candles holding. 65-84 = valid but weaker/choppier. Below 65 =
   HOLD, don't force it.

--- EXECUTION TARGET ({execution_ticker} - {multiplier}x leveraged) ---
* Live entry price: ${entry_price:.2f}

Write your full analysis in mathematical_proof before deciding action. Output
strict JSON matching the schema. Default to HOLD unless continuation is clean.
"""


def _build_fakeout_prompt(anchor, execution_ticker, fade_direction, multiplier, entry_price,
                           or_low, or_high, val, vah, level_price, ctx) -> str:
    return f"""
You are an expert algorithmic trading engine executing a LIQUIDITY-SWEEP
FAKEOUT FADE. Earlier this session, {anchor} broke the opening range
{"above its high" if fade_direction == "BEARISH" else "below its low"},
appearing to sweep a resting liquidity level near ${level_price:.2f} from the
prior session. Price has since reversed and a candle has closed back inside
the opening range's computed value area (${val:.2f}-${vah:.2f}), consistent
with a failed breakout / liquidity grab. The proposed trade is a
{fade_direction} move -- the OPPOSITE direction of the original breakout.
Judge whether this reversal looks genuine (momentum turning) or premature
(still just chop that happened to touch the value area once), and score
your confidence.

--- SETUP CONTEXT ({anchor}) ---
* Opening range               : ${or_low:.2f} - ${or_high:.2f}
* Suspected swept liquidity   : ${level_price:.2f}
* Value area (from OR volume) : ${val:.2f} - ${vah:.2f}
* Current anchor price        : ${ctx['latest_close']:.2f}
* Current time (ET)           : {ctx['current_time_str']}

--- RECENT 40-CANDLE MATRIX ({anchor}) ---
{ctx['recent_bars_json']}

--- YOUR TASK ---
1. Judge whether price action since the reversal supports genuine momentum
   shift or looks like weak, directionless chop.
2. Score confidence_score 1-100: 85-100 clean reversal. 65-84 valid but
   weaker. Below 65 = HOLD.
3. Stop-loss and take-profit are already fixed mechanically -- you are only
   judging the setup and confidence here, not setting prices.

--- EXECUTION TARGET ({execution_ticker} - {multiplier}x leveraged) ---
* Live entry price: ${entry_price:.2f}

Write your full analysis in mathematical_proof before deciding action. Output
strict JSON matching the schema. Default to HOLD unless the reversal looks genuine.
"""


async def evaluate_orb_setup(group_name: str, config: dict, ctx: dict, df_history: pd.DataFrame,
                              now: datetime.datetime, ny_tz):
    today = now.date()
    current_time = now.time()

    watch = _orb_fakeout_watch.get(group_name)
    if watch and watch["date"] == today:
        if current_time > FAKEOUT_DEADLINE:
            print(f"[ORB-Fakeout] {group_name}: past absolute deadline ({FAKEOUT_DEADLINE}). Clearing watch.")
            _orb_fakeout_watch.pop(group_name, None)
            _mark_orb_traded(group_name, today)
            return
    else:
        if current_time > ORB_DEADLINE:
            print(f"[ORB] {group_name}: past morning entry window ({ORB_DEADLINE}) -- skipping.")
            return

    trade_count = _orb_trade_count_today(group_name, today)
    if trade_count >= MAX_ORB_TRADES_PER_DAY:
        print(f"[ORB] {group_name}: already reached maximum of {MAX_ORB_TRADES_PER_DAY} ORB trades today ({trade_count}/{MAX_ORB_TRADES_PER_DAY}) -- skipping.")
        return

    anchor = config["anchor"]
    long_target = config["long_target"]
    short_target = config["short_target"]
    multiplier = config["leverage_multiplier"]
    session_start = now.replace(hour=9, minute=30, second=0, microsecond=0)

    # --- Path 1: active fakeout watch from a prior cycle ---
    watch = _orb_fakeout_watch.get(group_name)
    if watch and watch["date"] == today:
        elapsed_minutes = (now - watch["started_at"]).total_seconds() / 60
        if elapsed_minutes > FAKEOUT_WATCH_TIMEOUT_MINUTES:
            print(f"[ORB-Fakeout] {group_name}: watch timed out after {elapsed_minutes:.0f} min "
                  f"with no confirming reversal -- abandoning for the rest of today.")
            _orb_fakeout_watch.pop(group_name, None)
            _mark_orb_traded(group_name, today)
            return

        execution_ticker = short_target if watch["direction"] == "BULLISH" else long_target
        fade_direction = "BEARISH" if watch["direction"] == "BULLISH" else "BULLISH"
        val, vah = watch["val"], watch["vah"]

        try:
            target_start = datetime.datetime(now.year, now.month, now.day, 0, 0, tzinfo=ny_tz)
            bars_resp = await asyncio.to_thread(data_client.get_stock_bars, StockBarsRequest(
                symbol_or_symbols=execution_ticker, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=target_start, end=now, feed=DataFeed.IEX,
            ))
            tdf = bars_resp.df
            if isinstance(tdf.index, pd.MultiIndex):
                tdf = tdf.xs(execution_ticker)
            elif "timestamp" in tdf.columns:
                tdf = tdf.set_index("timestamp")
            tdf.index = pd.to_datetime(tdf.index)
            tdf.index = tdf.index.tz_localize("UTC").tz_convert(ny_tz) if tdf.index.tz is None else tdf.index.tz_convert(ny_tz)
        except Exception as e:
            print(f"⚠️ [ORB-Fakeout] Could not pull confirmation candle for {execution_ticker}: {e}")
            return

        if tdf.empty:
            return
        last_candle = tdf.iloc[-1]

        if not (val <= float(last_candle["close"]) <= vah):
            print(f"[ORB-Fakeout] {group_name}: still watching -- last close "
                  f"${float(last_candle['close']):.2f} outside value area ${val:.2f}-${vah:.2f}.")
            return

        entry_price = float(last_candle["close"])

        exec_or_end = session_start + datetime.timedelta(minutes=ORB_WINDOW_MINUTES)
        exec_or_bars = tdf[(tdf.index >= session_start) & (tdf.index < exec_or_end)]
        
        if exec_or_bars.empty:
            print(f"⚠️ [ORB-Fakeout] OR bars missing for {execution_ticker}")
            _orb_fakeout_watch.pop(group_name, None)
            return
            
        current_exec_or_high = float(exec_or_bars["high"].max())

        sl_price = round(float(last_candle["low"]), 2)
        tp_price = round(current_exec_or_high, 2)

        if sl_price >= entry_price or tp_price <= entry_price:
            print(f"⚠️ [ORB-Fakeout] Inverted bracket bounds for {execution_ticker} "
                  f"(Entry: ${entry_price}, SL: ${sl_price}, TP: ${tp_price}). Skipping.")
            _orb_fakeout_watch.pop(group_name, None)
            return

        prompt = _build_fakeout_prompt(anchor, execution_ticker, fade_direction, multiplier,
                                        entry_price, watch["or_low_exec"], watch["or_high_exec"],
                                        val, vah, watch["level_price"], ctx)
        try:
            response = await asyncio.to_thread(
                ai_client.models.generate_content,
                model="gemini-3.1-flash-lite", contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=ORBTradingSignal, temperature=0.1,
                ),
            )
            signal = json.loads(response.text)
        except Exception as e:
            print(f"⚠️ [ORB-Fakeout] Gemini call failed for {anchor}: {e}")
            return

        confidence = signal.get("confidence_score", 0)
        print(f"[ORB-Fakeout Parse] {anchor} fade={fade_direction}: confidence={confidence} -> {signal['action']}")

        if signal["action"] != "BUY" or confidence < 65:
            print(f"[ORB-Fakeout] {execution_ticker}: vetoed/low confidence ({confidence}) -- staying in watch.")
            return

        submitted = await _submit_orb_trade(
            group_name, execution_ticker, entry_price, sl_price, tp_price,
            confidence, signal["reasoning"], signal["mathematical_proof"],
            setup_type=f"ORB_FAKEOUT_FADE_{fade_direction}", anchor=anchor, ctx=ctx,
            extra_snapshot={"swept_level": watch["level_price"], "value_area_low": val, "value_area_high": vah},
        )
        if submitted:
            _orb_fakeout_watch.pop(group_name, None)
            _mark_orb_traded(group_name, today)
        return

    # --- Path 2: fresh breakout detection ---
    orb_range = get_opening_range(df_history, session_start, now)
    if orb_range is None:
        print(f"[ORB] {anchor}: opening range not ready yet or too narrow -- skipping this cycle.")
        return
    or_high, or_low, or_width, atr = orb_range

    direction = detect_orb_breakout(ctx["latest_close"], or_high, or_low)
    if direction is None:
        print(f"[ORB] {anchor}: no active breakout of the ${or_low:.2f}-${or_high:.2f} range.")
        return

    chase_distance = (ctx["latest_close"] - or_high) if direction == "BULLISH" else (or_low - ctx["latest_close"])
    if chase_distance > atr * MAX_CHASE_ATR:
        print(f"[ORB] {anchor}: too extended to chase, skipping.")
        return

    execution_ticker = long_target if direction == "BULLISH" else short_target

    try:
        target_start = datetime.datetime(now.year, now.month, now.day, 0, 0, tzinfo=ny_tz)
        target_bars_resp = await asyncio.to_thread(data_client.get_stock_bars, StockBarsRequest(
            symbol_or_symbols=execution_ticker, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=target_start, end=now, feed=DataFeed.IEX,
        ))
        target_df = target_bars_resp.df
        if isinstance(target_df.index, pd.MultiIndex):
            target_df = target_df.xs(execution_ticker)
        elif "timestamp" in target_df.columns:
            target_df = target_df.set_index("timestamp")
        target_df.index = pd.to_datetime(target_df.index)
        target_df.index = target_df.index.tz_localize("UTC").tz_convert(ny_tz) if target_df.index.tz is None else target_df.index.tz_convert(ny_tz)

        entry_price = float(target_df["close"].iloc[-1])
        exec_or_end = session_start + datetime.timedelta(minutes=ORB_WINDOW_MINUTES)
        exec_or_bars = target_df[(target_df.index >= session_start) & (target_df.index < exec_or_end)]
        if exec_or_bars.empty:
            print(f"⚠️ [ORB] Could not find opening-range bars for {execution_ticker}.")
            return
        exec_or_low = float(exec_or_bars["low"].min())
        exec_or_high = float(exec_or_bars["high"].max())
    except Exception as e:
        print(f"⚠️ [ORB] Could not pull execution pricing for {execution_ticker}: {e}")
        return
    
    MAX_EXEC_CHASE_PCT = 0.025  # Reject entry if ticker ran >2.5% past its 15-min OR High
    exec_chase_pct = (entry_price - exec_or_high) / exec_or_high
    if exec_chase_pct > MAX_EXEC_CHASE_PCT:
        print(f"[ORB] {execution_ticker} is +{exec_chase_pct * 100:.2f}% above OR High (max: {MAX_EXEC_CHASE_PCT * 100}%). Skipping overextended entry.")
        return

    # --- Liquidity-conflict check ---
    is_conflict, level_price = _detect_liquidity_conflict(direction, or_high, or_low, atr, ctx)
    if is_conflict:
        print(f"[ORB-Fakeout] {anchor}: {direction} breakout within {LIQUIDITY_PROXIMITY_ATR}x ATR "
              f"of prior-day level ${level_price:.2f} -- standing aside, arming fakeout watch.")
        try:
            minute_bars = await _fetch_minute_bars(execution_ticker, session_start, exec_or_end, ny_tz)
            profile = compute_value_area(minute_bars)
        except Exception as e:
            print(f"⚠️ [ORB-Fakeout] Could not compute volume profile for {execution_ticker}: {e}")
            return
        if profile is None:
            print(f"⚠️ [ORB-Fakeout] Value area computation failed for {execution_ticker}, skipping.")
            return

        val, vah, poc = profile
        _orb_fakeout_watch[group_name] = {
            "date": today, "direction": direction, "val": val, "vah": vah, "poc": poc,
            "or_high_exec": exec_or_high, "or_low_exec": exec_or_low,
            "level_price": level_price, "started_at": now,
        }
        print(f"[ORB-Fakeout] {group_name}: watch armed. Value area ${val:.2f}-${vah:.2f} "
              f"(POC ${poc:.2f}). Waiting up to {FAKEOUT_WATCH_TIMEOUT_MINUTES} min.")
        return

    # --- Immediate entry flow ---
    prompt = _build_orb_prompt(anchor, execution_ticker, direction, multiplier, entry_price,
                                or_high, or_low, atr, ctx)
    try:
        response = await asyncio.to_thread(
            ai_client.models.generate_content,
            model="gemini-3.1-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=ORBTradingSignal, temperature=0.1,
            ),
        )
        signal = json.loads(response.text)
    except Exception as e:
        print(f"⚠️ [ORB] Gemini call failed for {anchor}: {e}")
        return

    confidence = signal.get("confidence_score", 0)
    print(f"[ORB Parse] {anchor} {direction}: confidence={confidence} -> {signal['action']}")
    if signal["action"] != "BUY" or confidence < 65:
        print(f"[ORB] {execution_ticker}: vetoed or low confidence ({confidence}).")
        return

    # For all long ETF breakout entries (standard or inverse), structural support is the ETF's own Opening Range Low
    sl_price, tp_price = _calculate_orb_levels(entry_price, exec_or_low)
    if sl_price is None or tp_price is None:
        print(f"⚠️ [ORB] Entry price invalid vs OR boundary. Skipping.")
        return

    submitted = await _submit_orb_trade(
        group_name, execution_ticker, entry_price, sl_price, tp_price,
        confidence, signal["reasoning"], signal["mathematical_proof"],
        setup_type=f"ORB_{direction}", anchor=anchor, ctx=ctx,
        extra_snapshot={"opening_range_high": or_high, "opening_range_low": or_low},
    )
    if submitted:
        _mark_orb_traded(group_name, today)