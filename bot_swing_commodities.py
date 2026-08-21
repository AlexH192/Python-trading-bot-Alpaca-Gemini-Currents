import asyncio
import datetime
import enum
import json
import os
import zoneinfo

import pandas as pd
import numpy as np
import yfinance as yf

from typing import Optional

from pydantic import BaseModel, Field

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from google import genai
from google.genai import types

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.trading.requests import (
    MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest,
)
from alpaca.trading.stream import TradingStream

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, GEMINI_API_KEY, COMMODITY_MATRIX
from utils.telegram_notifier import TelegramNotifier
from utils.trade_logger import log_trade_to_journal

# =====================================================================
# CONFIGURATION
# =====================================================================
try:
    NY_TZ = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    NY_TZ = datetime.timezone(datetime.timedelta(hours=-5))

RISK_DOLLARS_PER_TRADE = 20.0   # $ risked per swing trade (stop distance x shares)
ENABLE_SHORT_SIDE = True         # set False to only take LONG snapback setups
DEFAULT_MAX_HOLD_DAYS = 5        # fallback if Gemini doesn't suggest one[cite: 2]

# New ATR Multiplier Settings
ATR_STOP_MULTIPLIER = 2.0        # Set Stop-Loss at 2x Daily ATR
ATR_TARGET_MULTIPLIER = 4.0      # Set Profit Target at 4x Daily ATR (1:2 Risk/Reward)

RECAP_LOOKBACK_DAYS = 45   # must cover the longest realistic max_hold_days so an
                           # entry from a prior day is still in the inventory-matching
                           # window when today's exit is being P&L-matched

ALL_SWING_TICKERS = []
for cfg in COMMODITY_MATRIX.values():
    ALL_SWING_TICKERS.extend([cfg["long_target"], cfg["short_target"]])


POSITION_STATE_PATH = "/Users/alex/Desktop/stock api python/New/BOT/Logs/swing_positions.json"
position_state_lock = asyncio.Lock()

data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trade_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
stream_client = TradingStream(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
ai_client = genai.Client(api_key=GEMINI_API_KEY)


# =====================================================================
# SCHEMA
# =====================================================================
class SwingAction(str, enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"

class SwingTradingSignal(BaseModel):
    wave_analysis: str = Field(
        description="Step-by-step read of the recent multi-day wave structure[cite: 2]."
    )
    action: SwingAction = Field(description="LONG if a bullish snapback confirms, SHORT if bearish, otherwise HOLD[cite: 2].")
    extension_confirmed: bool = Field(
        description="True only if price is meaningfully extended from its recent mean/range[cite: 2]."
    )
    overnight_alignment: bool = Field(
        description="True if the overnight futures move supports the proposed reversal[cite: 2]."
    )
    # entry_price, stop_loss, and take_profit have been removed. The bot calculates these mathematically.
    max_hold_days: int = Field(description="Suggested max trading days to hold if neither target nor stop is hit[cite: 2].")
    reasoning: str = Field(description="Brief explanation of the wave read and trade rationale[cite: 2].")


# =====================================================================
# POSITION STATE PERSISTENCE
# =====================================================================
async def _read_position_state() -> dict:
    def _sync_read():
        if not os.path.exists(POSITION_STATE_PATH):
            return {}
        with open(POSITION_STATE_PATH, "r") as f:
            content = f.read().strip()
            return json.loads(content) if content else {}
    return await asyncio.to_thread(_sync_read)

async def _write_position_state(state: dict):
    def _sync_write():
        os.makedirs(os.path.dirname(POSITION_STATE_PATH), exist_ok=True)
        with open(POSITION_STATE_PATH, "w") as f:
            json.dump(state, f, indent=4)
    await asyncio.to_thread(_sync_write)

async def record_new_swing_position(symbol: str, direction: str, entry_order_id: str, max_hold_days: int):
    async with position_state_lock:
        state = await _read_position_state()
        state[symbol] = {
            "direction": direction,
            "entry_date": datetime.date.today().isoformat(),
            "entry_order_id": entry_order_id,
            "max_hold_days": max_hold_days,
        }
        await _write_position_state(state)

async def clear_swing_position(symbol: str):
    async with position_state_lock:
        state = await _read_position_state()
        state.pop(symbol, None)
        await _write_position_state(state)


# =====================================================================
# DATA CONTEXT: OVERNIGHT FUTURES (yfinance)
# =====================================================================
def fetch_overnight_futures_context(yf_ticker: str) -> dict:
    try:
        tkr = yf.Ticker(yf_ticker)
        hourly = tkr.history(period="10d", interval="1h")
        daily = tkr.history(period="40d", interval="1d")
    except Exception as e:
        print(f"⚠️ yfinance fetch failed for {yf_ticker}: {e}")
        return {}

    if hourly.empty or daily.empty:
        return {}

    if hourly.index.tz is None:
        hourly.index = hourly.index.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        hourly.index = hourly.index.tz_convert(NY_TZ)

    now = datetime.datetime.now(NY_TZ)
    latest_price = float(hourly["Close"].iloc[-1])

    prior_close_cutoff = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now.hour < 16:
        prior_close_cutoff -= datetime.timedelta(days=1)
    prior_bars = hourly[hourly.index <= prior_close_cutoff]
    reference_price = float(prior_bars["Close"].iloc[-1]) if not prior_bars.empty else latest_price
    overnight_change_pct = (latest_price - reference_price) / reference_price if reference_price else 0.0

    sma10 = float(daily["Close"].tail(10).mean())
    sma20 = float(daily["Close"].tail(20).mean())
    dist_from_sma20_pct = (latest_price - sma20) / sma20 if sma20 else 0.0

    recent_high_20d = float(daily["High"].tail(20).max())
    recent_low_20d = float(daily["Low"].tail(20).min())

    daily_tr = (daily["High"] - daily["Low"]).tail(14)
    atr_14d = float(daily_tr.mean())

    recent_daily = daily.tail(15).copy().reset_index()
    recent_daily.columns = [str(c) for c in recent_daily.columns]
    date_col = "Date" if "Date" in recent_daily.columns else recent_daily.columns[0]
    recent_daily[date_col] = recent_daily[date_col].astype(str)
    recent_daily_json = recent_daily[[date_col, "Open", "High", "Low", "Close"]].to_json(orient="records")

    return {
        "yf_ticker": yf_ticker,
        "latest_futures_price": latest_price,
        "overnight_change_pct": overnight_change_pct,
        "sma10": sma10,
        "sma20": sma20,
        "dist_from_sma20_pct": dist_from_sma20_pct,
        "recent_high_20d": recent_high_20d,
        "recent_low_20d": recent_low_20d,
        "atr_14d": atr_14d,
        "recent_daily_json": recent_daily_json,
    }


# =====================================================================
# DATA CONTEXT: ETF HOURLY STRUCTURE (Alpaca)
# =====================================================================
def get_etf_swing_context(symbol: str) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(NY_TZ)
    start = now - datetime.timedelta(days=25) 

    try:
        bars = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            start=start, end=now, feed=DataFeed.IEX,
        )).df
    except Exception as e:
        print(f"⚠️ Failed to fetch 1H bars for {symbol}: {e}")
        return {"symbol": symbol, "has_data": False}

    if bars.empty:
        return {"symbol": symbol, "has_data": False}

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol)
    elif "timestamp" in bars.columns:
        bars = bars.set_index("timestamp")

    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        bars.index = bars.index.tz_convert(NY_TZ)

    bars = bars.between_time("09:30", "16:00")
    if bars.empty:
        return {"symbol": symbol, "has_data": False}

    latest_close = float(bars["close"].iloc[-1])
    today_bars = bars[bars.index.date == now.date()]
    today_open = float(today_bars["open"].iloc[0]) if not today_bars.empty else latest_close

    recent_high_10d = float(bars["high"].tail(70).max())   
    recent_low_10d = float(bars["low"].tail(70).min())

    recent_bars = bars.tail(60).copy()
    recent_bars["time_str"] = recent_bars.index.strftime("%Y-%m-%d %H:%M")
    recent_bars_json = recent_bars[["time_str", "open", "high", "low", "close", "volume"]].to_json(orient="records")

    return {
        "symbol": symbol,
        "has_data": True,
        "latest_close": latest_close,
        "today_open": today_open,
        "recent_high_10d": recent_high_10d,
        "recent_low_10d": recent_low_10d,
        "recent_bars_json": recent_bars_json,
        "current_time_str": now.strftime("%Y-%m-%d %H:%M"),
    }

def get_latest_price(symbol: str) -> Optional[float]:
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(NY_TZ)
    start = now - datetime.timedelta(minutes=30)
    try:
        bars = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start, end=now, feed=DataFeed.IEX,
        )).df
    except Exception as e:
        print(f"⚠️ Failed to fetch latest price for {symbol}: {e}")
        return None

    if bars.empty:
        return None
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol)
    return float(bars["close"].iloc[-1])


# =====================================================================
# NEW: EXECUTION TICKER ATR CALCULATION
# =====================================================================
def get_etf_atr(symbol: str, period: int = 14) -> float:
    """Calculates the 14-period Daily Average True Range (ATR) directly on the execution ETF."""
    now = datetime.datetime.now(NY_TZ)
    start = now - datetime.timedelta(days=40) 
    try:
        bars = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
            start=start, end=now, feed=DataFeed.IEX
        )).df
        if bars.empty: return 0.0
        
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol)
            
        # True Range Calculation
        bars['prev_close'] = bars['close'].shift(1)
        bars['tr1'] = bars['high'] - bars['low']
        bars['tr2'] = (bars['high'] - bars['prev_close']).abs()
        bars['tr3'] = (bars['low'] - bars['prev_close']).abs()
        bars['tr'] = bars[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        return float(bars['tr'].tail(period).mean())
    except Exception as e:
        print(f"⚠️ ATR fetch failed for {symbol}: {e}")
        return 0.0


# =====================================================================
# GEMINI PROMPT
# =====================================================================
def build_swing_prompt(name: str, etf: str, futures_ctx: dict, etf_ctx: dict) -> str:
    return f"""
You are an expert commodities swing trader. Your strategy is based on the idea
that commodities like gold and crude oil move in multi-day waves: a strong
directional leg (impulse) tends to overextend, then "snaps back" toward its
recent mean before continuing or reversing further[cite: 2].

You are evaluating {name} via its tradable ETF ({etf}), but informed by the
corresponding futures market ({futures_ctx['yf_ticker']})[cite: 2].

--- OVERNIGHT / FUTURES CONTEXT ({futures_ctx['yf_ticker']}) ---
* Latest futures price                  : ${futures_ctx['latest_futures_price']:.2f}
* Overnight change (since last US close): {futures_ctx['overnight_change_pct']:.2%}
* 10-day SMA                            : ${futures_ctx['sma10']:.2f}
* 20-day SMA                            : ${futures_ctx['sma20']:.2f}
* Distance from 20-day SMA              : {futures_ctx['dist_from_sma20_pct']:.2%}
* 20-day High / Low                     : ${futures_ctx['recent_high_20d']:.2f} / ${futures_ctx['recent_low_20d']:.2f}
* 14-day ATR (proxy, daily true range)  : ${futures_ctx['atr_14d']:.2f}

--- RECENT DAILY FUTURES CANDLES (last ~15 sessions) ---
{futures_ctx['recent_daily_json']}

--- ETF INTRADAY STRUCTURE ({etf}, 1-hour candles) ---
* Current time (ET)      : {etf_ctx['current_time_str']}
* Latest close           : ${etf_ctx['latest_close']:.2f}
* Today's open           : ${etf_ctx['today_open']:.2f}
* ~10-trading-day High/Low: ${etf_ctx['recent_high_10d']:.2f} / ${etf_ctx['recent_low_10d']:.2f}

--- RECENT 1-HOUR CANDLES ({etf}) ---
{etf_ctx['recent_bars_json']}

--- YOUR TASK ---
1. Read the wave structure across the daily futures candles[cite: 2].
2. Only mark extension_confirmed = true if the move is genuinely stretched[cite: 2].
3. Check overnight_alignment: does the futures overnight move support a reversal thesis[cite: 2].
4. If both confirm, decide LONG (expecting a bounce off an oversold extension) or SHORT (expecting a pullback from an overbought extension). Otherwise HOLD.
5. Suggest max_hold_days: how long you'd give this trade to work before invalidating the thesis on time (typically 2-7 trading days)[cite: 2].

Write your full analysis in wave_analysis BEFORE deciding action[cite: 2]. Output strict JSON matching the required schema[cite: 2]. Default to HOLD unless the setup is clean[cite: 2].
"""


# =====================================================================
# RISK SIZING & ORDER EXECUTION
# =====================================================================
def calculate_shares(entry_price: float, stop_loss: float, risk_dollars: float = RISK_DOLLARS_PER_TRADE) -> int:
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0:
        return 0
    return max(1, int(risk_dollars / risk_per_share))

async def submit_swing_entry(name: str, cfg: dict, action: SwingAction, signal: dict):
    if action == SwingAction.LONG:
        execution_ticker = cfg["long_target"]
    else:  
        execution_ticker = cfg["short_target"]

    exec_entry_price = get_latest_price(execution_ticker)
    if exec_entry_price is None:
        print(f"⚠️ Could not get a live price for {execution_ticker}, skipping entry.")
        return None

    # Fetch ETF-specific ATR for dynamic risk sizing
    atr = await asyncio.to_thread(get_etf_atr, execution_ticker)
    if atr == 0.0:
        print(f"⚠️ Could not calculate ATR for {execution_ticker}, skipping entry.")
        return None

    # Since both LONG and SHORT theses buy the asset (Regular or Inverse ETF), we ALWAYS place stops below and targets above.
    exec_stop = exec_entry_price - (atr * ATR_STOP_MULTIPLIER)
    exec_target = exec_entry_price + (atr * ATR_TARGET_MULTIPLIER)

    shares = calculate_shares(exec_entry_price, exec_stop)
    if shares < 1:
        print(f"Risk sizing produced 0 shares for {execution_ticker}, skipping.")
        return None

    order_req = MarketOrderRequest(
        symbol=execution_ticker, qty=shares, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(exec_target, 2)),
        stop_loss=StopLossRequest(stop_price=round(exec_stop, 2)),
    )

    try:
        order = await asyncio.to_thread(trade_client.submit_order, order_data=order_req)
    except Exception as e:
        print(f"⚠️ Order submission failed for {execution_ticker}: {e}")
        return None

    max_hold_days = signal.get("max_hold_days") or DEFAULT_MAX_HOLD_DAYS
    await record_new_swing_position(execution_ticker, action.value, str(order.id), max_hold_days)

    await log_trade_to_journal(
        ticker=execution_ticker,
        action=f"SWING_ENTRY_{action.value}",
        fill_price=exec_entry_price,
        quantity=shares,
        reasoning=signal["reasoning"],
        market_snapshot={
            "wave_analysis": signal["wave_analysis"],
            "anchor_etf": cfg["anchor_etf"],
            "atr_value_used": round(atr, 2),
            "exec_stop": round(exec_stop, 2), "exec_target": round(exec_target, 2),
            "max_hold_days": max_hold_days,
        },
    )

    await TelegramNotifier.send(
        f"*SWING ENTRY: {action.value}*\n"
        f"* Anchor: {cfg['anchor_etf']} | Execution: {execution_ticker}\n"
        f"* Shares: {shares}\n"
        f"* Entry ~${exec_entry_price:.2f}\n"
        f"* Stop: ${exec_stop:.2f} (-{ATR_STOP_MULTIPLIER}x ATR) | Target: ${exec_target:.2f} (+{ATR_TARGET_MULTIPLIER}x ATR)\n"
        f"* Max hold: {max_hold_days} trading days\n\n"
        f"*Rationale*: {signal['reasoning']}"
    )
    return order


# =====================================================================
# MAX-HOLD-DAY EXIT CHECK
# =====================================================================
async def check_max_hold_exits():
    state = await _read_position_state()
    if not state:
        return

    today = datetime.date.today()
    for symbol, info in list(state.items()):
        entry_date = datetime.date.fromisoformat(info["entry_date"])
        trading_days_held = int(np.busday_count(entry_date, today)) 

        if trading_days_held >= info["max_hold_days"]:
            try:
                open_orders = await asyncio.to_thread(
                    trade_client.get_orders,
                    filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]),
                )
                for o in open_orders:
                    await asyncio.to_thread(trade_client.cancel_order_by_id, o.id)
                await asyncio.sleep(1.0)
                await asyncio.to_thread(trade_client.close_position, symbol)

                await TelegramNotifier.send(
                    f"⏱ *SWING TIME EXIT*\n"
                    f"* Symbol: {symbol}\n"
                    f"* Held {trading_days_held} trading days (max {info['max_hold_days']}). Closed at market."
                )
                await log_trade_to_journal(
                    ticker=symbol, action="SWING_TIME_EXIT", fill_price=None, quantity=None,
                    reasoning="Max hold days reached without hitting target or stop.",
                )
                await clear_swing_position(symbol)
            except Exception as e:
                print(f"⚠️ Max-hold exit failed for {symbol}: {e}")


# =====================================================================
# OVERNIGHT / PRE-MARKET BRIEFING
# =====================================================================
async def send_overnight_briefing():
    print("Generating overnight commodities briefing...")
    sections = []
    for name, cfg in COMMODITY_MATRIX.items():
        ctx = await asyncio.to_thread(fetch_overnight_futures_context, cfg["futures_yf_ticker"])
        if not ctx:
            continue
        sections.append(
            f"{name} ({cfg['futures_yf_ticker']}): overnight {ctx['overnight_change_pct']:+.2%}, "
            f"{ctx['dist_from_sma20_pct']:+.2%} from 20d SMA."
        )
    if not sections:
        return

    prompt = f"""
    Using this overnight futures data for gold and crude oil, write a concise pre-market briefing[cite: 2]:

    {chr(10).join(sections)}

    Layout:
    *OVERNIGHT COMMODITIES BRIEFING*
    * Gold: [1-2 sentences]
    * Crude Oil: [1-2 sentences]
    * Swing Watch: [note if either looks like it may be approaching an extension worth watching today]

    Objective tone, no emojis, no filler[cite: 2].
    """
    try:
        response = await asyncio.to_thread(
            ai_client.models.generate_content,
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        await TelegramNotifier.send(response.text)
    except Exception as e:
        print(f"⚠️ Overnight briefing failed: {e}")

# =====================================================================
# DAILY RECAP / P&L
# =====================================================================
async def send_daily_swing_recap():
    print("Compiling daily swing recap...")
    today = datetime.date.today()
    lookback_start = datetime.datetime.now(NY_TZ) - datetime.timedelta(days=RECAP_LOOKBACK_DAYS)

    try:
        params = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=lookback_start,
            symbols=ALL_SWING_TICKERS,
            limit=500,
        )
        orders = await asyncio.to_thread(trade_client.get_orders, filter=params)

        valid_orders = [o for o in orders
                         if o.filled_at is not None and o.filled_avg_price is not None and o.qty is not None]
        sorted_orders = sorted(valid_orders, key=lambda o: o.filled_at)

        inventory: dict[str, list[dict]] = {}
        closed_today = []

        for o in sorted_orders:
            qty = float(o.qty)
            price = float(o.filled_avg_price)
            symbol = o.symbol
            filled_date = o.filled_at.astimezone(NY_TZ).date()

            if o.side == OrderSide.BUY:
                inventory.setdefault(symbol, []).append(
                    {"price": price, "qty": qty, "filled_date": filled_date}
                )

            elif o.side == OrderSide.SELL:
                sell_proceeds, buy_cost, remaining = 0.0, 0.0, qty
                matched_entry_dates = []

                while remaining > 0 and inventory.get(symbol):
                    lot = inventory[symbol][0]
                    match_qty = min(remaining, lot["qty"])
                    buy_cost += match_qty * lot["price"]
                    sell_proceeds += match_qty * price
                    matched_entry_dates.append(lot["filled_date"])
                    lot["qty"] -= match_qty
                    remaining -= match_qty
                    if lot["qty"] <= 0:
                        inventory[symbol].pop(0)

                if filled_date == today:
                    hold_days = int(np.busday_count(min(matched_entry_dates), filled_date)) if matched_entry_dates else None
                    closed_today.append({
                        "symbol": symbol, "qty": qty, "exit_price": price,
                        "pnl": sell_proceeds - buy_cost, "hold_days": hold_days,
                    })

        total_trades = len(closed_today)
        total_pnl = sum(t["pnl"] for t in closed_today)
        winners = sum(1 for t in closed_today if t["pnl"] > 0)
        win_rate = (winners / total_trades * 100) if total_trades else 0.0

        try:
            open_positions = await asyncio.to_thread(trade_client.get_all_positions)
            still_open = [p.symbol for p in open_positions if p.symbol in ALL_SWING_TICKERS]
        except Exception:
            still_open = []

        summary = (
            f"*DAILY SWING PERFORMANCE SUMMARY*\n\n"
            f"* Trades Closed Today: {total_trades}\n"
            f"* Win Rate: {win_rate:.1f}%\n"
            f"* Realized P&L: ${total_pnl:.2f}\n"
            f"* Currently Open Swing Positions: {len(still_open)}\n\n"
        )

        if closed_today:
            summary += "*Closed Today:*\n"
            for t in closed_today:
                hold_str = f", held {t['hold_days']}d" if t["hold_days"] is not None else ""
                summary += f"* {t['symbol']}: {t['qty']:.0f} sh @ ${t['exit_price']:.2f} | P&L: ${t['pnl']:.2f}{hold_str}\n"
        else:
            summary += "_No swing positions closed today._\n"

        if still_open:
            summary += f"\n*Still Open:* {', '.join(still_open)}"

        await TelegramNotifier.send(summary)

    except Exception as e:
        print(f"Error compiling daily swing recap: {e}")
        await TelegramNotifier.send(f"⚠️ Failed to compile daily swing recap: {e}")


# =====================================================================
# TRADE UPDATE STREAM HANDLER
# =====================================================================
async def on_trade_update(data):
    event_str = getattr(data.event, "value", str(data.event)).lower()
    if event_str not in ("fill", "partial_fill"):
        return

    order = data.order
    symbol = order.symbol
    order_type_str = str(order.type).lower()
    parent_id = str(getattr(order, "parent_order_id", ""))

    if not parent_id or parent_id == "None":
        return 

    if "limit" in order_type_str:
        await TelegramNotifier.send(f"🎯 *SWING TARGET HIT*\n* Symbol: {symbol}\n* Snapback target reached.")
        await log_trade_to_journal(
            ticker=symbol, action="SWING_TP", fill_price=order.filled_avg_price,
            quantity=order.filled_qty, reasoning="Snapback take-profit target hit.",
        )
        await clear_swing_position(symbol)
    elif "stop" in order_type_str:
        await TelegramNotifier.send(f"🛡 *SWING STOP HIT*\n* Symbol: {symbol}\n* Position stopped out.")
        await log_trade_to_journal(
            ticker=symbol, action="SWING_SL", fill_price=order.filled_avg_price,
            quantity=order.filled_qty, reasoning="Stop-loss triggered.",
        )
        await clear_swing_position(symbol)


# =====================================================================
# CORE HOURLY CYCLE
# =====================================================================
async def run_swing_cycle():
    now = datetime.datetime.now(NY_TZ)
    print(f"\n==== Swing cycle at {now.strftime('%Y-%m-%d %H:%M:%S')} ET ====")

    try:
        open_positions = await asyncio.to_thread(trade_client.get_all_positions)
        held_symbols = {p.symbol for p in open_positions}
    except Exception as e:
        print(f"⚠️ Error reading positions: {e}")
        held_symbols = set()

    for name, cfg in COMMODITY_MATRIX.items():
        anchor = cfg["anchor_etf"]
        yf_ticker = cfg["futures_yf_ticker"]

        if cfg["long_target"] in held_symbols or cfg["short_target"] in held_symbols:
            print(f"{name} already has an open swing position -- skipping new-entry scan.")
            continue

        futures_ctx = await asyncio.to_thread(fetch_overnight_futures_context, yf_ticker)
        etf_ctx = await asyncio.to_thread(get_etf_swing_context, anchor)

        if not futures_ctx or not etf_ctx.get("has_data"):
            print(f"Skipping {name}: insufficient data this cycle.")
            continue

        prompt = build_swing_prompt(name, anchor, futures_ctx, etf_ctx)

        try:
            response = await asyncio.to_thread(
                ai_client.models.generate_content,
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SwingTradingSignal,
                    temperature=0.15,
                ),
            )
            signal = json.loads(response.text)
        except Exception as e:
            print(f"⚠️ Gemini call failed for {name}: {e}")
            continue

        print(f"[{anchor}] Verdict: {signal['action']} | Extension: {signal['extension_confirmed']} "
              f"| Overnight aligned: {signal['overnight_alignment']}")

        if signal["action"] == "HOLD" or not signal["extension_confirmed"] or not signal["overnight_alignment"]:
            print(f"No confirmed snapback setup for {name} this cycle.")
            continue

        action_enum = SwingAction(signal["action"])
        if action_enum == SwingAction.SHORT and not ENABLE_SHORT_SIDE:
            print(f"SHORT signal for {name} suppressed (ENABLE_SHORT_SIDE = False).")
            continue

        await submit_swing_entry(name, cfg, action_enum, signal)


# =====================================================================
# MAIN
# =====================================================================
async def main():
    scheduler = AsyncIOScheduler()

    scheduler.add_job(run_swing_cycle, 'cron', day_of_week='mon-fri', hour='9-15', minute=35, timezone='US/Eastern')
    scheduler.add_job(send_overnight_briefing, 'cron', day_of_week='mon-fri', hour=9, minute=33, timezone='US/Eastern')
    scheduler.add_job(check_max_hold_exits, 'cron', day_of_week='mon-fri', hour=9, minute=35, timezone='US/Eastern')
    scheduler.add_job(send_daily_swing_recap, 'cron', day_of_week='mon-fri', hour=16, minute=5, timezone='US/Eastern')

    scheduler.start()
    stream_client.subscribe_trade_updates(on_trade_update)

    all_tickers = []
    for cfg in COMMODITY_MATRIX.values():
        all_tickers.extend([cfg["long_target"], cfg["short_target"]])
    print("====================================================")
    print("Commodities swing bot initialized.")
    print(f"Tracking: {', '.join(all_tickers)} on hourly cycles.")
    print("====================================================")

    await TelegramNotifier.send(
        f"Commodities swing bot initialized. Monitoring {', '.join(all_tickers)} on hourly cycles."
    )

    print("Running initial boot validation cycle...")
    await run_swing_cycle()

    print("\nStarting continuous streaming runtime")
    await asyncio.to_thread(stream_client.run)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nStopping swing bot smoothly. Goodbye!")
        try:
            stream_client.stop_ws()
        except Exception:
            pass
        print("Stopping swing bot smoothly. Goodbye!")