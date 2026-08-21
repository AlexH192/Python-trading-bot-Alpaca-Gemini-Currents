"""
Enriches trade journal JSON based on market info at timestamp
Usage:
    python3 '/Users/alex/Desktop/stock api python/New/BOT/Logs/enrich_trade_journal.py' --reconciled '/Users/alex/Desktop/stock api python/New/BOT/Logs/reconciled_trades.json' --out enriched_trades.json
"""

import argparse
import datetime
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from utils.alpaca_client import data_client

try:
    from strategies.regime_detector import compute_regime_metrics, resolve_regime
    HAVE_REGIME = True
except ImportError:
    HAVE_REGIME = False


import zoneinfo

NY_TZ = zoneinfo.ZoneInfo("America/New_York")


import zoneinfo

NY_TZ = zoneinfo.ZoneInfo("America/New_York")
LEGACY_SERVER_TZ = zoneinfo.ZoneInfo("Europe/Berlin") # CET/CEST with DST handled automatically


def _to_ny_timestamp(raw_ts) -> tuple[pd.Timestamp, bool]:
    ts = pd.Timestamp(raw_ts)
    if ts.tzinfo is None:
        return ts.tz_localize(LEGACY_SERVER_TZ).tz_convert(NY_TZ), True
    return ts.tz_convert(NY_TZ), False


def _fetch_bars(ticker: str, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    resp = data_client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=ticker, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start, end=end, feed=DataFeed.IEX,
    ))
    df = resp.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(NY_TZ)
    return df.sort_index()


def _mfe_mae(bars: pd.DataFrame, entry_price: float, direction: str, risk_per_share: float):
    """direction: 'long' or 'short'. risk_per_share must be > 0."""
    if bars.empty or risk_per_share <= 0:
        return None, None
    if direction == "long":
        mfe = (bars["high"].max() - entry_price) / risk_per_share
        mae = (entry_price - bars["low"].min()) / risk_per_share
    else:
        mfe = (entry_price - bars["low"].min()) / risk_per_share
        mae = (bars["high"].max() - entry_price) / risk_per_share
    return round(float(mfe), 2), round(float(mae), 2)


def _target_reached_after_exit(bars_after_exit: pd.DataFrame, target: float, direction: str) -> bool:
    if bars_after_exit.empty or target is None:
        return False
    if direction == "long":
        return bool((bars_after_exit["high"] >= target).any())
    return bool((bars_after_exit["low"] <= target).any())


def enrich_trade(trade: dict) -> dict:
    ticker = trade["ticker"]
    entry_price = trade["entry_price"]

    entry_ts, entry_tz_assumed = _to_ny_timestamp(trade["entry_ts"])
    exit_ts, exit_tz_assumed = _to_ny_timestamp(trade.get("exit_ts", trade["entry_ts"]))
    if entry_tz_assumed or exit_tz_assumed:
        trade["timezone_assumed"] = True

    direction = "long" if trade["strategy"] in ("SWEEP", "ORB") else trade.get("direction", "long")
    stop = trade.get("stop_loss")
    targets = [t for t in (trade.get("tp1"), trade.get("tp2"), trade.get("take_profit")) if t is not None]
    risk_per_share = abs(entry_price - stop) if stop is not None else None

    session_end = entry_ts.normalize() + pd.Timedelta(hours=16)

    if entry_ts >= session_end:
        trade["enrichment_error"] = (
            f"entry_ts ({entry_ts}) falls after computed session_end ({session_end}) "
            f"-- timestamp likely misparsed or outside normal session hours"
        )
        return trade
    if exit_ts < entry_ts:
        trade["enrichment_error"] = f"exit_ts ({exit_ts}) precedes entry_ts ({entry_ts}) -- check reconcile.py pairing"
        return trade

    try:
        bars = _fetch_bars(ticker, entry_ts.to_pydatetime(), session_end.to_pydatetime())
    except Exception as e:
        trade["enrichment_error"] = f"bar fetch failed: {e}"
        return trade

    trade_bars = bars[(bars.index >= entry_ts) & (bars.index <= exit_ts)]
    post_exit_bars = bars[bars.index > exit_ts]

    if risk_per_share:
        mfe, mae = _mfe_mae(trade_bars, entry_price, direction, risk_per_share)
        trade["mfe_r"] = mfe
        trade["mae_r"] = mae
        trade["exit_r"] = round(trade["pnl"] / (risk_per_share * trade["exit_qty"]), 2) if trade["exit_qty"] else None

    for label, target in zip(("tp1", "tp2"), targets):
        if target is not None:
            trade[f"{label}_reached_after_exit"] = _target_reached_after_exit(post_exit_bars, target, direction)

    trade["time_of_day"] = entry_ts.strftime("%H:%M")
    trade["day_of_week"] = entry_ts.strftime("%A")

    if HAVE_REGIME:
        try:
            session_start = entry_ts.normalize() + pd.Timedelta(hours=9, minutes=30)
            window = bars[(bars.index >= entry_ts - pd.Timedelta(days=4)) & (bars.index <= entry_ts)]
            metrics = compute_regime_metrics(window, session_start, None, None)
            if metrics is not None:
                trade["regime_at_entry"] = resolve_regime(ticker, metrics).label
        except Exception as e:
            trade["regime_lookup_error"] = str(e)

    return trade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconciled", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trades = json.load(open(args.reconciled))
    enriched = [enrich_trade(t) for t in trades if abs(t.get("open_qty", 1)) < 0.01]  # closed trades only

    json.dump(enriched, open(args.out, "w"), indent=2, default=str)
    print(f"Enriched {len(enriched)} closed trades -> {args.out}")


if __name__ == "__main__":
    main()