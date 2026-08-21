import datetime
import json
import time
import zoneinfo
import pandas as pd
import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from google.genai import types

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderClass, QueryOrderStatus, ActivityType
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest, StopOrderRequest, ReplaceOrderRequest, ClosePositionRequest

# Module Imports
from config.settings import WATCHLIST_MATRIX, ALL_EXECUTION_TICKERS, RISK_DOLLARS_PER_TRADE, MIN_TP1_RR, MIN_SL_DISTANCE_PCT, MAX_NOTIONAL_PER_TRADE
from utils.alpaca_client import trade_client, data_client, stream_client
from utils.telegram_notifier import TelegramNotifier
from utils.trade_logger import log_trade_to_journal
from utils.market_data import fetch_finance_news, get_market_structure_context
from strategies.gemini_reasoning import ai_client, AdvancedTradingSignal
from utils.split_order_tracker import *
from strategies.regime_detector import compute_regime_metrics, resolve_regime, compute_early_window_signal
from strategies.orb_strategy import evaluate_orb_setup

# Global State Dictionary to link our independent split targets
last_liquidation_date = None
RISK_DOLLARS_PER_TRADE = 10.0   # $ risked per trade, split across TP1/TP2 legs for A-Tier setups
MIN_SL_DISTANCE_PCT = 0.80   # SL must be at least this fraction of the baseline
                               # scaled_risk_pct away from entry -- prevents Gemini
                               # from picking an unrealistically tight stop that
                               # blows up position size under fixed-dollar-risk sizing

# --- Event Deduplication Tracking ---
_processed_order_events: set[str] = set()
_notified_order_ids: set[str] = set()

_regime_history_cache: dict[str, pd.DataFrame] = {}

async def get_regime_history(anchor: str, now: datetime.datetime) -> pd.DataFrame:
    """Rolling multi-day 5-min bar cache per anchor -- fetches only new
    bars each cycle instead of the full lookback window every time."""
    cached = _regime_history_cache.get(anchor)
    fetch_start = (cached.index[-1] + datetime.timedelta(seconds=1)) if cached is not None and not cached.empty \
        else (now - datetime.timedelta(days=6))

    new_df = pd.DataFrame()
    if fetch_start < now:
        bars = await asyncio.to_thread(data_client.get_stock_bars, StockBarsRequest(
            symbol_or_symbols=anchor, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=fetch_start, end=now, feed=DataFeed.IEX,
        ))
        new_df = bars.df
        if isinstance(new_df.index, pd.MultiIndex):
            new_df = new_df.xs(anchor)
        new_df = new_df.sort_index()

    combined = pd.concat([cached, new_df]) if cached is not None else new_df
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined[combined.index >= now - datetime.timedelta(days=4)]

    _regime_history_cache[anchor] = combined
    return combined


def calculate_shares(entry_price: float, stop_loss: float, risk_dollars: float,
                      max_notional: float = MAX_NOTIONAL_PER_TRADE) -> int:

    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0:
        return 0
    risk_based_qty = max(1, int(risk_dollars / risk_per_share))
    notional_capped_qty = max(1, int(max_notional / entry_price))
    return min(risk_based_qty, notional_capped_qty)

def _extract_bracket_legs(order):
    legs = getattr(order, "legs", None) or []
    tp_leg = next((l for l in legs if str(getattr(l, "type", "")).lower() == "limit"), None)
    sl_leg = next((l for l in legs if str(getattr(l, "type", "")).lower() == "stop"), None)
    return tp_leg, sl_leg

async def send_morning_briefing():
    print("Generating Morning Briefing...")
    
    anchors = [cfg["anchor"] for cfg in WATCHLIST_MATRIX.values()]
    market_news = await fetch_finance_news(anchors)
    
    prompt = f"""
    Using the following financial news headlines, provide a concise market briefing:
    
    {market_news}
    
    Market Context: Broader indices ({', '.join(anchors)}).
    
    Layout:
    *MORNING MARKET BRIEFING*
    * Macro Catalysts: [Summarize the news above in 2-4 bullets]
    * Market Sentiment: [Sentiment score (good/neutral/bad) + Context based on the news provided]
    * Strategy Outlook: [Detail how our liquidity-sweep/BOS/FVG strategy should behave, in 1-2 bullet points]
    
    Do not use emojis or conversational filler. Keep it objective, but not too complicated.
    """

    max_retries = 3
    delay = 2 
    async_client = ai_client.aio 

    for attempt in range(max_retries):
        try:
            response = await async_client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            
            await TelegramNotifier.send(response.text)
            return

        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                print(f"Rate limited (Attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2  
            else:
                print(f"An unexpected error occurred: {e}")
                break

async def send_closing_summary():
    print("Compiling Closing Summary...")
    try:
        ny_tz = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        ny_tz = datetime.timezone(datetime.timedelta(hours=-5))
        
    today_start = datetime.datetime.now(ny_tz).replace(hour=0, minute=0, second=0, microsecond=0)
    
    try:
        params = GetOrdersRequest(status='closed', after=today_start, limit=100)
        filled_orders = await asyncio.to_thread(trade_client.get_orders, filter=params)
        
        valid_orders = [o for o in filled_orders if o.filled_at is not None]
        sorted_orders = sorted(valid_orders, key=lambda o: o.filled_at)
        
        total_pnl = 0.0
        total_trades = 0
        winning_trades = 0
        details = []
        inventory = {}

        for o in sorted_orders:
            if o.filled_avg_price is None or o.qty is None:
                continue
            
            qty = float(o.qty)
            price = float(o.filled_avg_price)
            symbol = o.symbol

            if o.side == OrderSide.BUY:
                if symbol not in inventory: inventory[symbol] = []
                inventory[symbol].append({'price': price, 'qty': qty})
            
            elif o.side == OrderSide.SELL:
                sell_proceeds = 0.0
                buy_cost = 0.0
                remaining_sell_qty = qty
                
                while remaining_sell_qty > 0 and symbol in inventory and len(inventory[symbol]) > 0:
                    buy_lot = inventory[symbol][0]
                    match_qty = min(remaining_sell_qty, buy_lot['qty'])
                    
                    buy_cost += (match_qty * buy_lot['price'])
                    sell_proceeds += (match_qty * price)
                    
                    buy_lot['qty'] -= match_qty
                    remaining_sell_qty -= match_qty
                    if buy_lot['qty'] <= 0:
                        inventory[symbol].pop(0)

                trade_pnl = sell_proceeds - buy_cost
                total_pnl += trade_pnl
                
                total_trades += 1
                if trade_pnl > 0: winning_trades += 1
                details.append(f"* {symbol} Closed: {qty} shares | P&L: ${trade_pnl:.2f}")

        success_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        summary_msg = f"*DAILY PERFORMANCE SUMMARY*\n\n"
        summary_msg += f"* Total Closed Trades: {total_trades}\n"
        summary_msg += f"* Win Rate: {success_rate:.1f}%\n"
        summary_msg += f"* Realized Daily P&L: ${total_pnl:.2f}\n" 
        summary_msg += f"* Portfolio Status: Reconciled\n\n"
        
        if details:
            summary_msg += "*Closed Transactions:*\n" + "\n".join(details)
        else:
            summary_msg += "_No transactions finalized during today's session._"
            
        await TelegramNotifier.send(summary_msg)
    except Exception as e:
        print(f"Error compiling daily trading metrics: {e}")

async def move_twin_to_breakeven(symbol: str, twin_sl_id: str, entry_price: float):
    if twin_sl_id == "NONE":
        return
    try:
        existing_order = await asyncio.to_thread(trade_client.get_order_by_id, twin_sl_id)
        original_qty = existing_order.qty

        await asyncio.to_thread(trade_client.cancel_order_by_id, twin_sl_id)

        new_stop_request = StopOrderRequest(
            symbol=symbol,
            qty=original_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            stop_price=round(entry_price, 2),
        )
        new_order = await asyncio.to_thread(trade_client.submit_order, order_data=new_stop_request)
        await rekey_split_order_leg(old_id=twin_sl_id, new_id=str(new_order.id))
        await set_breakeven_flag(str(new_order.id))

        await TelegramNotifier.send(
            f"🛡️ *RISK LOCK ADVANCEMENT*\n* Asset: {symbol}\n"
            f"* Modification: Runner Stop Loss updated cleanly to Entry Cost: ${entry_price:.2f}."
        )
    except Exception as e:
        print(f"⚠️ Error modifying twin stop loss contract metrics: {e}")
        await TelegramNotifier.send(
            f"⚠️ *RISK LOCK FAILED* for {symbol} -- runner position may currently have "
            f"NO protective stop live. Check manually: {e}"
        )

async def on_trade_update(data):
    event_str = getattr(data.event, 'value', str(data.event)).lower()
    if event_str not in ('fill', 'partial_fill'):
        return

    order = data.order
    order_id = str(order.id)
    order_type_str = str(order.type).lower()

    # --- STRICT DEDUPLICATION FOR LIMIT ORDERS (TP) ---
    if 'limit' in order_type_str:
        if order_id in _notified_order_ids:
            return  # Silently ignore subsequent batch/partial updates for this order
        _notified_order_ids.add(order_id)

    filled_qty = getattr(order, 'filled_qty', '0')

    # --- EXISTING EVENT DEDUPLICATION CHECK ---
    dedup_key = f"{order_id}_{event_str}_{filled_qty}"
    if dedup_key in _processed_order_events:
        return
    _processed_order_events.add(dedup_key)

    ticker = order.symbol
    side_str = getattr(order.side, 'value', str(order.side)).lower()
    order_type_str = str(order.type).lower()

    raw_parent_id = getattr(order, 'parent_id', None)
    parent_id = str(raw_parent_id) if raw_parent_id else None
    has_parent = bool(parent_id) and parent_id != 'None'

    leg_info = await get_split_order_leg(order_id)
    if leg_info is None:
        for _ in range(6):
            await asyncio.sleep(0.5)
            leg_info = await get_split_order_leg(order_id)
            if leg_info is not None:
                break

    matched_parent_id = parent_id if has_parent else order_id

    if has_parent:
        leg_info = await get_split_order_leg(parent_id)
        if leg_info is None:
            for _ in range(6):
                await asyncio.sleep(0.5)
                leg_info = await get_split_order_leg(parent_id)
                if leg_info is not None:
                    break

    if leg_info is None and side_str == 'sell':
        match = await find_leg_by_ticker(ticker)
        if match:
            matched_parent_id, leg_info = match

    if leg_info is None:
        if side_str == 'sell' and 'market' in order_type_str:
            print(f"⚠️ Liquidation fill for {ticker} with no matching tracked entry.")
            await log_trade_to_journal(
                ticker=ticker,
                action="LIQUIDATION",
                fill_price=order.filled_avg_price,
                quantity=order.filled_qty,
                reasoning="Time stop liquidation. No tracked entry found.",
                market_snapshot={}
            )
        return

    # --- Classify by ORDER TYPE, not by parent presence ---
    if 'market' in order_type_str:
        await TelegramNotifier.send(
            f"🛡️ *STRATEGY DISPATCH: POSITION LIQUIDATED*\n"
            f"* Asset Target: {ticker}\n"
            f"* Fill Price: ${float(order.filled_avg_price):.2f}\n"
            f"* Status: Time-stop liquidation filled."
        )
        await log_trade_to_journal(
            ticker=ticker,
            action="LIQUIDATION",
            fill_price=order.filled_avg_price,
            quantity=order.filled_qty,
            reasoning="Time stop threshold reached (3:55 PM restriction).",
            market_snapshot={"entry_price": leg_info["entry_price"]}
        )
        twin_id = leg_info.get('twin_id', 'NONE')
        twin_sl_id = leg_info.get('twin_sl_id', 'NONE')
        await clear_split_order_pair(matched_parent_id, twin_id, twin_sl_id)
        return

    if 'limit' in order_type_str:
        leg_type = leg_info.get('type', '')
        twin_id = leg_info.get('twin_id', 'NONE')
        twin_sl_id = leg_info.get('twin_sl_id', 'NONE')

        # --- Handle Single-Target ORB Trades Cleanly ---
        if leg_type in ('ORB_SINGLE', 'TP_SINGLE') or twin_id == 'ORB_SINGLE':
            await TelegramNotifier.send(
                f"🎯 *STRATEGY DISPATCH: TP TARGET CLEARED*\n"
                f"* Asset Target: {ticker}\n"
                f"* Status: ORB profit target reached."
            )
            await log_trade_to_journal(
                ticker=ticker,
                action="SELL_TP",
                fill_price=order.filled_avg_price,
                quantity=order.filled_qty,
                reasoning="ORB target hit.",
                market_snapshot={"entry_price": leg_info["entry_price"]}
            )
            # Clear tracking memory for this single target -- its own SL leg too
            if twin_id not in ("NONE", "ORB_SINGLE"):
                await clear_split_order_pair(matched_parent_id, twin_id, twin_sl_id)
            else:
                await clear_split_order_pair(matched_parent_id)

        elif leg_type == 'TP1_LEG':
            await TelegramNotifier.send(
                f"🎯 *STRATEGY DISPATCH: TP1 ACHIEVED*\n"
                f"* Asset Target: {ticker}\n"
                f"* Action: Scaled out Initial Target\n"
                f"* Status: Tightening secondary risk profiles to Breakeven..."
            )
            await log_trade_to_journal(
                ticker=ticker,
                action="SELL_TP1",
                fill_price=order.filled_avg_price,
                quantity=order.filled_qty,
                reasoning="TP1 runner target hit.",
                market_snapshot={"entry_price": leg_info["entry_price"]}
            )
            if twin_sl_id != "NONE":
                await move_twin_to_breakeven(ticker, twin_sl_id, leg_info['entry_price'])

        elif leg_type == 'TP2_LEG':
            await TelegramNotifier.send(
                f"🏆 *STRATEGY DISPATCH: TP2 TARGET CLEARED*\n"
                f"* Asset Target: {ticker}\n"
                f"* Status: Final 50% runner hit structural expansion target."
            )
            await log_trade_to_journal(
                ticker=ticker,
                action="SELL_TP2",
                fill_price=order.filled_avg_price,
                quantity=order.filled_qty,
                reasoning="TP2 runner target hit.",
                market_snapshot={"entry_price": leg_info["entry_price"]}
            )
            if twin_id != "NONE":
                await clear_split_order_pair(matched_parent_id, twin_id, twin_sl_id)

    elif 'stop' in order_type_str:
        is_be = leg_info.get('is_breakeven', False)
        context_str = "Runner stopped out at initial Entry Cost floor." if is_be else "Protective Initial Stop Loss executed."

        await TelegramNotifier.send(
            f"🛡️ *STRATEGY DISPATCH: POSITION STOPOUT*\n"
            f"* Asset Target: {ticker}\n"
            f"* Context: {context_str}\n"
            f"* Status: Strategy cycle terminated."
        )
        await log_trade_to_journal(
            ticker=ticker,
            action="SELL_STOP_BE" if is_be else "SELL_STOP_INITIAL",
            fill_price=order.filled_avg_price,
            quantity=order.filled_qty,
            reasoning=context_str,
            market_snapshot={
                "entry_price": leg_info["entry_price"],
                "leg_type": leg_info["type"],
                "was_breakeven": is_be
            }
        )
        twin_id = leg_info.get('twin_id', 'NONE')
        twin_sl_id = leg_info.get('twin_sl_id', 'NONE')
        if twin_id != "NONE":
            await clear_split_order_pair(matched_parent_id, twin_id, twin_sl_id)

async def run_trading_cycle():
    try:
        ny_tz = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        ny_tz = datetime.timezone(datetime.timedelta(hours=-5))
        
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(ny_tz)
    timestamp_now = now.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n====================================================")
    print(f"TIMER TRIGGERED: Running execution cycle at {timestamp_now}")
    print(f"====================================================")
    
    if (now.hour == 15 and now.minute >= 55) or now.hour >= 16:
        global last_liquidation_date
        current_date = datetime.date.today()
        if last_liquidation_date != current_date:
            print("TIME STOP MATCHED (3:55 PM or later). Bot-managed liquidation sequence engaged...")
            
            try:
                active_positions = await asyncio.to_thread(trade_client.get_all_positions)
            except Exception as e:
                print(f"⚠️ Error fetching positions: {e}")
                active_positions = []
            
            bot_positions = [p for position in active_positions if (p := position).symbol in ALL_EXECUTION_TICKERS]
        
            if not bot_positions:
                print("No active bot-managed positions found. Liquidation skipped.")
            else:
                for position in bot_positions:
                    ticker = position.symbol
                    try:
                        print(f"Closing position and cancelling open bracket orders for {ticker}...")
                        
                        # 1. Fetch open bracket orders for this specific ticker
                        order_filter = GetOrdersRequest(
                            status=QueryOrderStatus.OPEN,
                            symbols=[ticker]
                        )
                        open_orders = await asyncio.to_thread(trade_client.get_orders, filter=order_filter)
                        
                        # 2. Cancel the open orders via their Order IDs
                        for order in open_orders:
                            await asyncio.to_thread(trade_client.cancel_order_by_id, order_id=order.id)
                            
                        # 3. Prevent API Race Condition
                        # A brief pause prevents a known race condition where Alpaca's backend 
                        # might still momentarily register the shares as "held" by the canceled orders.
                        await asyncio.sleep(1)
                            
                        # 4. Create a valid ClosePositionRequest for 100% of the position
                        close_request = ClosePositionRequest(percentage="100")
                        
                        # 5. Submit position closure
                        await asyncio.to_thread(
                            trade_client.close_position, 
                            symbol_or_asset_id=ticker,
                            close_options=close_request,
                        )
                        
                    except Exception as e:
                        print(f"Error liquidating {ticker}: {e}")
                        await TelegramNotifier.send(f"⚠️ Liquidation error for {ticker}: {e}")
                        
            last_liquidation_date = current_date
            print("Bot liquidation sequence completed. Flag set to prevent re-triggering.")
        else:
            print("Liquidation already performed for today. Skipping.")
        return
            
    try:
        open_positions = await asyncio.to_thread(trade_client.get_all_positions)
        currently_held_tickers = [position.symbol for position in open_positions]
        print(f"Account's current asset holdings: {currently_held_tickers}")
    except Exception as e:
        print(f"⚠️ Error reading portfolio metrics from account: {e}")
        currently_held_tickers = []

    is_after_1030 = (now.hour > 10 or (now.hour == 10 and now.minute >= 30))

    for group_name, config in WATCHLIST_MATRIX.items():
        anchor = config["anchor"]
        long_target = config["long_target"]
        short_target = config["short_target"]
        multiplier = config["leverage_multiplier"]
        
        execution_ticker = None
        setup_direction = None
        base_breakout_pct = 0.0
        
        print(f"\nEvaluating advanced liquidity matrix rules for Group: {group_name} (Anchor: {anchor})")
        
        if (long_target in currently_held_tickers) or (short_target in currently_held_tickers):
            print(f"Position already active within {group_name}. Skipping asset scan search.")
            continue
            
        ctx = get_market_structure_context(anchor)
        try:
            bar_count = len(json.loads(ctx['recent_bars_json']))
            print(f"DEBUG: Context for {anchor} returned {bar_count} bars.")
        except Exception:
            print("DEBUG: Could not parse JSON for bar count.")
        
        try:
            bars_data = json.loads(ctx['recent_bars_json'])
            df = pd.DataFrame(bars_data)
            
            time_cols = ['time_str', 't', 'timestamp', 'time', 'start', 'date']
            time_col = next((c for c in df.columns if c in time_cols), None)
            
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col], format='mixed')
                df.set_index(time_col, inplace=True)
            
            print(f"📊 Processing {len(df)} candles for {anchor}:")
            print(f"{'Time':<20} | {'Open':<8} | {'High':<8} | {'Low':<8} | {'Close':<8}")
            print("-" * 65)
            
            for idx, row in df.tail(5).iterrows():
                time_str = idx.strftime('%H:%M:%S') 
                print(f"{time_str:<20} | {row['open']:<8.2f} {row['high']:<8.2f} {row['low']:<8.2f} {row['close']:<8.2f}")
        
        except Exception as e:
            print(f"⚠️ Error parsing or displaying data: {e}")
            df = None

        try:
            df_history = await get_regime_history(anchor, now)
            session_start_ts = now.replace(hour=9, minute=30, second=0, microsecond=0)

            metrics = compute_regime_metrics(df_history, session_start_ts, ctx['first_hour_high'], ctx['first_hour_low'])
            if metrics is not None:
                regime_state = resolve_regime(anchor, metrics)
                active_regime = regime_state.label
                print(f"[Regime] {anchor}: raw={regime_state.raw_label} active={active_regime} "
                    f"confirmed={regime_state.confirmed} (ADX={metrics.adx:.1f}, "
                    f"VWAP-dist={metrics.dist_from_vwap_atr:.2f}ATR, containment={metrics.range_containment:.0%})")
            else:
                early = compute_early_window_signal(df_history, session_start_ts)
                active_regime = "TRENDING" if (early and early.candidate_trending) else "RANGING"
                if early:
                    print(f"[Regime-Early] {anchor}: gap={early.gap_atr:.2f}ATR "
                        f"opening_range={early.opening_bar_range_atr:.2f}ATR -> {active_regime}")
        except Exception as e:
            print(f"⚠️ Regime detection failed for {anchor}, defaulting to RANGING: {e}")
            active_regime = "RANGING"

        if active_regime == "TRENDING":
            print(f"{anchor} regime=TRENDING — running ORB strategy instead.")
            await evaluate_orb_setup(group_name, config, ctx, df_history, now, ny_tz)
            continue

        if active_regime == "TRANSITIONAL":
            print(f"{anchor} regime=TRANSITIONAL — standing aside this cycle.")
            continue


        if ctx['latest_close'] < ctx['prev_day_low']:
            execution_ticker = long_target 
            setup_direction = "BULLISH_LONG_SWEEP"
            base_breakout_pct = (ctx['prev_day_low'] - ctx['latest_close']) / ctx['prev_day_low']
        elif ctx['latest_close'] > ctx['prev_day_high']:
            execution_ticker = short_target 
            setup_direction = "BEARISH_SHORT_SWEEP"
            base_breakout_pct = (ctx['latest_close'] - ctx['prev_day_high']) / ctx['prev_day_high']
        
        if not execution_ticker and is_after_1030:
            if ctx['latest_close'] < ctx['first_hour_low']:
                execution_ticker = long_target
                setup_direction = "FIRST_HOUR_LOW_SWEEP"
                base_breakout_pct = (ctx['first_hour_low'] - ctx['latest_close']) / ctx['first_hour_low']
            elif ctx['latest_close'] > ctx['first_hour_high']:
                execution_ticker = short_target
                setup_direction = "FIRST_HOUR_HIGH_SWEEP"
                base_breakout_pct = (ctx['latest_close'] - ctx['first_hour_high']) / ctx['first_hour_high']

        if not execution_ticker:
            print(f"Anchor asset {anchor} remains within active strategy boundaries. No trade triggered.")
            continue
            
        try:
            target_start = datetime.datetime(now.year, now.month, now.day, 0, 0, tzinfo=ny_tz)
            target_bars = await asyncio.to_thread(data_client.get_stock_bars, StockBarsRequest(
                symbol_or_symbols=execution_ticker, timeframe=TimeFrame(5, TimeFrameUnit.Minute), start=target_start, end=now, feed=DataFeed.IEX
            ))
            
            target_df = target_bars.df
            if isinstance(target_df.index, pd.MultiIndex): 
                target_df = target_df.xs(execution_ticker)
            elif 'timestamp' in target_df.columns:
                target_df = target_df.set_index('timestamp')
                
            target_current_price = float(target_df['close'].iloc[-1])
        except Exception as e:
            print(f"⚠️ Could not pull real-time execution statistics: {e}")
            continue

        baseline_risk_pct = 0.005  
        baseline_reward_pct = 0.012 
        
        scaled_risk_pct = baseline_risk_pct * multiplier
        scaled_reward_pct = baseline_reward_pct * multiplier
        
        calculated_stop_loss = target_current_price * (1.0 - scaled_risk_pct)
        calculated_take_profit = target_current_price * (1.0 + scaled_reward_pct)
        
        prompt = f"""
        You are an expert algorithmic trading engine executing a multi-timeframe liquidity sweep strategy. You analyze an anchor asset (e.g., QQQ, SPY) to determine precise long entry signals for its corresponding 3x leveraged vehicles (TQQQ/SQQQ or SPXL/SPXS).
        Review the provided real-time structural constants and evaluate the following strategy parameters exactly. You must only issue a BUY action if all confirmation rules match perfectly; otherwise, output a HOLD.

        --- CORE STRUCTURAL BOUNDARIES ---
        1. You are provided with verified liquidity levels: Yesterday's High (PDH), Yesterday's Low (PDL), and the Opening Range (ORB). 
        2. These boundaries are hard thresholds. You are not tasked with identifying them; you are tasked with validating the trade setup based on their breach.
        3. Your primary directive is to confirm if the price action occurring at these specific boundaries displays the required structural integrity (BOS + FVG) to justify an entry.
        4. Hard Time Restriction: Between 09:30 AM EST and 10:30 AM EST, you must ONLY look for breaks of the primary boundaries (PDH/PDL). Do not evaluate secondary boundaries during this opening hour.
        5. Opening Range Fallback: If the current time is AFTER 10:30 AM EST and the anchor asset has failed to break PDH or PDL, you are now authorized to activate the secondary fallback: the 1-hour Opening Range. This range is defined by the absolute High and Low prices printed during the first hour of today's session (09:30 AM - 10:30 AM EST).
        6. Once you see the anchor asset break past a legally active boundary (PDH/PDL before 10:30 AM, or either set after 10:30 AM), you must immediately evaluate the provided 40-candle intraday history context for strategy validation.

        --- 40-CANDLE INTRADAY MATRIX HISTORY ---
        You are being passed a sequential array of the last forty 5-minute candles of the anchor index. You must act as your own chart analysis engine to evaluate macro/micro trend structure:
        * Locate key structural macro swing highs and swing lows generated throughout the entire session.
        * Verify structural confirmations over a broad time arc, avoiding noise spikes.

        --- EXECUTION & CONFIRMATION CRITERIA ---
        • BULLISH LONG SETUP (Set action to BUY for the Long Target ETF, e.g., TQQQ):
          Triggers when the anchor asset breaks BELOW Yesterday's Low (or the 1-hour Opening Range Low after 10:30 AM). 
          Validation Process: 
          1. Scan the 40-candle history matrix. Identify a significant swing high point printed earlier.
          2. Check if a recent 5-minute candle's CLOSE value (not a wick) broke cleanly ABOVE that swing high (Break of Structure - BOS).
          3. Confirm that this strong move created a Bullish Fair Value Gap (FVG)—a 3-candle sequence where Candle 3's Low remains completely higher than Candle 1's High, leaving an open valuation imbalance void.
          4. Confirm that the current price at the end of the candle matrix has retraced down inside that specific FVG zone. If it hasn't re-entered the gap yet, hold.

        • BEARISH SHORT SETUP (Set action to BUY for the Short Target ETF, e.g., SQQQ):
          Triggers when the anchor asset breaks ABOVE Yesterday's High (or the 1-hour Opening Range High after 10:30 AM). 
          Validation Process:
          1. Scan the 40-candle history matrix. Identify a significant swing low point printed earlier.
          2. Check if a recent 5-minute candle's CLOSE value (not a wick) broke cleanly BELOW that swing low (Break of Structure - BOS).
          3. Confirm that this strong move created a Bearish Fair Value Gap (FVG)—a 3-candle sequence where Candle 3's High remains completely lower than Candle 1's Low, leaving an open valuation imbalance void.
          4. Confirm that the current price at the end of the candle matrix has retraced up inside that specific FVG zone. If it hasn't re-entered the gap yet, hold.

        --- RISK & POSITION MANAGEMENT RULES ---

        1. Initial Stop Loss: You are provided a 'Volatility-Scaled SL' baseline by the system. Validate that this level is positioned safely behind the recent structural low/high of the liquidity sweep. If it is too tight or too wide, suggest an adjustment in your reasoning.
        2. Partial Profit Taking (TP1): Set TP1 at a major local liquidity level or swing boundary that provides at least a 0.5R to 1.0R Risk-to-Reward ratio relative to entry and Stop Loss. Do NOT place TP1 at shallow micro-FVG bounds or minor wicks that are less than 0.5R away from entry.
        3. Final Target (TP2): The system targets a 'Volatility-Scaled TP Baseline'. Evaluate this target against the opposing daily boundary (e.g., PDH for long trades). Ensure the target is reachable within the session and that the projected R:R (Risk-to-Reward) ratio remains above 2.5.

        --- REQUIRED CHAIN OF THOUGHT (mathematical_proof) ---
        You must write out your step-by-step logic in the `mathematical_proof` field BEFORE deciding the action.

        For BULLISH LONG SETUPS, you must explicitly write:
        1. "SWING HIGH IDENTIFICATION: The swing high is at [Timestamp] with a high of $[Price]."
        2. "BOS EVALUATION: The candle at [Timestamp] closed at $[Price], which is [greater than / less than] the swing high of $[Price]. BOS is [Valid/Invalid]."
        3. "FVG EVALUATION: Looking at the 3-candle sequence ending at [Timestamp]: Candle 1 High = $[Price], Candle 3 Low = $[Price]. Gap = $[Price] to $[Price]. FVG is [Valid/Invalid]."
        4. "RETEST EVALUATION: The current price is $[Price]. This [is / is not] inside the FVG zone. Retest is [Valid/Invalid]."

        For BEARISH SHORT SETUPS, invert the math (look for Swing Lows, Candle 1 Low vs Candle 3 High).

        --- DYNAMIC SCORING RULES ---
        You must assign a `confidence_score` (1-100) to the setup:
        * A-Tier (85-100): Textbook sweep, massive displacement candle breaking structure, and a deep, clean retest into the FVG.
        * B-Tier (65-84): Valid sweep and BOS, but the displacement is weaker, or the FVG is exceptionally narrow. 
        * C-Tier (Below 65): Messy price action, unconvincing BOS, or poor R:R metrics. Output action as HOLD.
        
        --- STRATEGY MATRIX PARAMETERS ---
        • Base Anchor Index          : {anchor}
        • Execution Asset Target     : {execution_ticker}
        • Direction Strategy Path    : {setup_direction}
        • Anchor Penetration Degree  : {base_breakout_pct:.2%}

        --- CURRENT REAL-TIME STRUCTURAL CONSTANTS FOR ANCHOR ({anchor}) ---
        • Current Price right now : ${ctx['latest_close']:.2f}
        • Current New York Time   : {ctx['current_time_str']} EST
        • Today's Opening Price   : ${ctx['today_open']:.2f}
        • Previous Day's High     : ${ctx['prev_day_high']:.2f}
        • Previous Day's Low      : ${ctx['prev_day_low']:.2f}
        • Today's First-Hour High : ${ctx['first_hour_high']:.2f}
        • Today's First-Hour Low  : ${ctx['first_hour_low']:.2f}
        ---------------------------------------------------------

        --- RAW CANDLE OHLC MATRIX DATA (LAST 40 CANDLES) ---
        {ctx['recent_bars_json']}
        ---------------------------------------------------------

        --- EXECUTION TARGET PRICING ({execution_ticker} - {multiplier}x Leveraged) ---
        • Live Target Entry Price : ${target_current_price:.2f}
        • Volatility-Scaled SL Baseline: ${calculated_stop_loss:.2f} (-{scaled_risk_pct:.1%})
        • Volatility-Scaled TP Baseline: ${calculated_take_profit:.2f} (+{scaled_reward_pct:.1%})
        
        --- OUTPUT INSTRUCTIONS ---
        Analyze the provided matrix parameters, verify all multi-candle structural confirmations visually through the data table and through your chain of thought, and output a strict JSON configuration matching the requested application/json schema definition exactly. Evaluate the conditions strictly. If ANY of the BOS, FVG, or Retest evaluations fail, you MUST output a HOLD action.
        """
        
        try:
            response = await asyncio.to_thread(ai_client.models.generate_content,
                model='gemini-3.1-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AdvancedTradingSignal,
                    temperature=0.1,
                ),
            )
            
            signal = json.loads(response.text)
            action = signal["action"]
            reasoning = signal["reasoning"]
            confidence = signal.get("confidence_score", 0)
            
            print(f"[Gemini Strategy Parse] Setup: {signal['setup_type']} | BOS: {signal['bos_confirmed']} | FVG Entry: {signal['fvg_entered']} | Score: {confidence}")
            print(f"👉 Verdict: {action} -> {reasoning}")
            
            if action == "BUY" and signal["bos_confirmed"] and signal["fvg_entered"]:
                sl_price = round(signal["calculated_stop_loss"], 2)
                tp1_price = round(signal["calculated_tp1"], 2)
                tp2_price = round(signal["calculated_tp2"], 2)

                min_risk_distance = target_current_price * scaled_risk_pct * MIN_SL_DISTANCE_PCT
                current_risk_distance = target_current_price - sl_price
                if current_risk_distance < min_risk_distance:
                    print(f"⚠️ Model SL (${sl_price:.2f}) was too tight "
                        f"({current_risk_distance:.3f} < {min_risk_distance:.3f} floor). Widening.")
                    sl_price = round(target_current_price - min_risk_distance, 2)

                # --- TP1 MINIMUM RISK-TO-REWARD GUARDRAIL ---
                risk_distance = target_current_price - sl_price
                if risk_distance > 0:
                    min_tp1_price = round(target_current_price + (risk_distance * MIN_TP1_RR), 2)
                    if tp1_price < min_tp1_price:
                        print(f"⚠️ Model TP1 (${tp1_price:.2f}) was too tight (<{MIN_TP1_RR}R). Clamping to ${min_tp1_price:.2f}.")
                        tp1_price = min_tp1_price

                # Ensure TP2 remains higher than TP1
                if tp2_price <= tp1_price:
                    tp2_price = round(tp1_price + (risk_distance * 0.75), 2)
                    print(f"⚠️ Model TP2 was below/equal to TP1. Clamping TP2 to ${tp2_price:.2f}.")

                max_reasonable_tp2 = target_current_price * (1.0 + scaled_reward_pct * 1.15)
                if tp2_price > max_reasonable_tp2:
                    print(f"⚠️ Model TP2 ({tp2_price}) exceeded sane cap ({max_reasonable_tp2:.2f}), clamping.")
                    tp2_price = round(max_reasonable_tp2, 2)

                # --- DYNAMIC TIERED SIZING EXECUTION ---
                if confidence >= 85:
                    # A-Tier: split the full risk budget evenly across both legs, since
                    # both legs share the same entry/stop distance.
                    leg_risk_dollars = RISK_DOLLARS_PER_TRADE / 2
                    qty1 = calculate_shares(target_current_price, sl_price, leg_risk_dollars)
                    qty2 = calculate_shares(target_current_price, sl_price, leg_risk_dollars)

                    if qty1 == 0 or qty2 == 0:
                        print(f"⚠️ Invalid stop distance (SL == entry?) for {execution_ticker}. Skipping trade.")
                        continue

                    print(f"🌟 A-Tier Setup Detected (Score: {confidence}). "
                          f"Sizing to ~${RISK_DOLLARS_PER_TRADE:.2f} risk: {qty1}+{qty2} shares.")

                    buy_request_tp1 = MarketOrderRequest(
                        symbol=execution_ticker, qty=qty1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
                        take_profit=TakeProfitRequest(limit_price=tp1_price),
                        stop_loss=StopLossRequest(stop_price=sl_price)
                    )
                    buy_request_tp2 = MarketOrderRequest(
                        symbol=execution_ticker, qty=qty2, side=OrderSide.BUY, time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
                        take_profit=TakeProfitRequest(limit_price=tp2_price),
                        stop_loss=StopLossRequest(stop_price=sl_price)
                    )

                    order1 = await asyncio.to_thread(trade_client.submit_order, order_data=buy_request_tp1)
                    order2 = await asyncio.to_thread(trade_client.submit_order, order_data=buy_request_tp2)

                    tp1_leg, sl1_leg = _extract_bracket_legs(order1)
                    tp2_leg, sl2_leg = _extract_bracket_legs(order2)

                    if not all([tp1_leg, sl1_leg, tp2_leg, sl2_leg]):
                        # .legs sometimes isn't populated inline on the submit response --
                        # re-fetch to get the child leg ids before giving up.
                        order1 = await asyncio.to_thread(trade_client.get_order_by_id, order1.id)
                        order2 = await asyncio.to_thread(trade_client.get_order_by_id, order2.id)
                        tp1_leg, sl1_leg = _extract_bracket_legs(order1)
                        tp2_leg, sl2_leg = _extract_bracket_legs(order2)

                    if not all([tp1_leg, sl1_leg, tp2_leg, sl2_leg]):
                        print(f"⚠️ Could not resolve bracket child leg ids for {execution_ticker}. "
                              f"Position is live but UNTRACKED -- monitor manually.")
                        continue

                    await register_split_order_pair(
                        tp1_id=str(tp1_leg.id), sl1_id=str(sl1_leg.id),
                        tp2_id=str(tp2_leg.id), sl2_id=str(sl2_leg.id),
                        entry_price=target_current_price, ticker=execution_ticker,
                    )
                    print(f"Split entry submission complete. Linked TP leg IDs: {tp1_leg.id} & {tp2_leg.id}")

                    logged_qty = qty1 + qty2
                    tier_str = f"A-Tier ({qty1}+{qty2} Shares Split)"

                elif 65 <= confidence < 85:
                    qty = calculate_shares(target_current_price, sl_price, RISK_DOLLARS_PER_TRADE)

                    if qty == 0:
                        print(f"⚠️ Invalid stop distance (SL == entry?) for {execution_ticker}. Skipping trade.")
                        continue

                    print(f"⚖️ B-Tier Setup Detected (Score: {confidence}). "
                        f"Sizing to ~${RISK_DOLLARS_PER_TRADE:.2f} risk: {qty} shares (TP1 Only).")

                    buy_request_tp1 = MarketOrderRequest(
                        symbol=execution_ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
                        take_profit=TakeProfitRequest(limit_price=tp1_price),
                        stop_loss=StopLossRequest(stop_price=sl_price)
                    )

                    order1 = await asyncio.to_thread(trade_client.submit_order, order_data=buy_request_tp1)

                    tp1_leg, sl1_leg = _extract_bracket_legs(order1)

                    if not all([tp1_leg, sl1_leg]):
                        # Same fix as A-Tier -- .legs isn't always populated inline on the
                        # submit response, so re-fetch before giving up.
                        order1 = await asyncio.to_thread(trade_client.get_order_by_id, order1.id)
                        tp1_leg, sl1_leg = _extract_bracket_legs(order1)

                    if not all([tp1_leg, sl1_leg]):
                        print(f"⚠️ Could not resolve bracket child leg ids for {execution_ticker}. "
                            f"Position is live but UNTRACKED -- monitor manually.")
                        await TelegramNotifier.send(
                            f"⚠️ *UNTRACKED POSITION WARNING*\n"
                            f"* Asset: {execution_ticker}\n"
                            f"* A B-Tier bracket order filled, but leg ids could not be resolved. "
                            f"Position is live on Alpaca but not tracked internally -- check manually."
                        )
                        continue

                    await register_single_order(str(tp1_leg.id), str(sl1_leg.id), target_current_price, execution_ticker)

                    print(f"Standalone bracket entry submission complete. Linked ID: {order1.id}")

                    logged_qty = qty
                    tier_str = f"B-Tier ({qty} Share{'s' if qty != 1 else ''} TP1 Scalp)"

                else:
                    print(f"⚠️ C-Tier Setup (Score: {confidence}). Structure lacks strong confirmation. Skipping execution.")
                    continue

                await log_trade_to_journal(
                    ticker=execution_ticker,
                    action="BUY",
                    fill_price=target_current_price,
                    quantity=logged_qty,
                    reasoning=reasoning,
                    market_snapshot={
                        "setup_type": signal["setup_type"],
                        "confidence_score": confidence,
                        "anchor": anchor,
                        "anchor_price": ctx["latest_close"],
                        "stop_loss": sl_price,
                        "tp1": tp1_price,
                        "tp2": tp2_price,
                        "mathematical_proof": signal["mathematical_proof"],
                    }
                )
                
                await TelegramNotifier.send(
                    f" *STRATEGY ORDER EXECUTED* \n\n"
                    f"* Ticker Asset : {execution_ticker}\n"
                    f"* Strategy Path: {group_name} ({signal['setup_type']})\n"
                    f"* Conviction   : {tier_str} | Score: {confidence}\n"
                    f"* Entry Execution: ${target_current_price:.2f}\n"
                    f"* Initial Protective SL: ${sl_price:.2f}\n\n"
                    f"🎯 *Scale Target (TP1)*: ${tp1_price:.2f}\n"
                    + (f"🏆 *Runner Target (TP2)*: ${tp2_price:.2f}\n\n" if confidence >= 85 else "\n") +
                    f"*AI Strategy Rationale*: {reasoning}"
                )
            else:
                print(f"Setup conditions incomplete for {execution_ticker}. No positions initiated.")
                
        except Exception as e:
            print(f"Error processing strategy loop calculations: {e}")

async def main():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(run_trading_cycle, 'cron', minute='*/5', second=5)
    scheduler.add_job(send_morning_briefing, 'cron', day_of_week='mon-fri', hour=9, minute=35, timezone='US/Eastern')
    scheduler.add_job(send_closing_summary, 'cron', day_of_week='mon-fri', hour=16, minute=0, timezone='US/Eastern')
    scheduler.start()
    
    stream_client.subscribe_trade_updates(on_trade_update)
    
    all_anchors = [cfg["anchor"] for cfg in WATCHLIST_MATRIX.values()]
    print("====================================================")
    print("Trading bot initialization successful.")
    print(f"Monitoring Anchors: {', '.join(all_anchors)} via Gemini 3.1 Flash-Lite.")
    print("WebSocket persistent order channel streaming live.")
    print("====================================================")
    
    await TelegramNotifier.send(f"Intraday liquidity sweep bot initialized. Monitoring {len(WATCHLIST_MATRIX)} base tickers via WebSocket stream loop.")
    
    print("Running initial boot validation cycle...")
    await run_trading_cycle()
    
    print("\nStarting continuous streaming runtime")
    await asyncio.to_thread(stream_client.run)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nStopping algorithm daemon smoothly. Goodbye!")
        try:
            print("Disconnecting Alpaca stream...")
            stream_client.stop_ws()
        except Exception as e:
            pass 
            
        print("Stopping algorithm daemon smoothly. Goodbye!")