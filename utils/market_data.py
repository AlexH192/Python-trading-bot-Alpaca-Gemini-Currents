import pandas as pd
import datetime
import zoneinfo
import httpx
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from utils.alpaca_client import data_client
from config.settings import CURRENTS_API_KEY
import os


async def fetch_finance_news(anchors: list[str]) -> str:
    """Fetches recent financial news headlines for the monitored market anchors via Currents API (v1)."""
    
    # 1. Broadened query: Removed strict quotes from single words
    anchor_query = " OR ".join([f'"{a}"' for a in anchors]) if anchors else '"QQQ" OR "SPY"'
    macro_keywords = 'economy OR "Federal Reserve" OR inflation OR "interest rates" OR GDP'
    full_query = f"({anchor_query}) OR ({macro_keywords})"

    # 2. Strict UTC Time Formatting
    # APIs parse pure UTC 'Z' (Zulu time) much more reliably than localized offsets
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    yesterday_utc = now_utc - datetime.timedelta(days=1)
    start_date_str = yesterday_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    params = {
        "query": full_query,
        "language": "en",
        # Dropped the 'category' filter to prevent artificial throttling on the API side
        "page_size": 10,         
        "start_date": start_date_str
    }
    
    headers = {
        "Authorization": f"Bearer {CURRENTS_API_KEY}"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.currentsapi.services/v1/search", 
                params=params, 
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"⚠️ Currents API Error {response.status_code}: {response.text}")
                
            response.raise_for_status()
            data = response.json()
            
            news_items = data.get("news", [])
            if not news_items:
                return "No recent financial news articles found for current anchors in the last 24 hours."

            # Format top 5 headlines for the morning briefing prompt
            headlines = [f"• {item.get('title')}" for item in news_items[:5]]
            return "\n".join(headlines)

    except httpx.HTTPStatusError as e:
        print(f"⚠️ HTTP Error fetching news: {e}")
        return "Financial news stream currently unavailable."
    except Exception as e:
        print(f"⚠️ Error fetching news from Currents API: {e}")
        return "Financial news stream currently unavailable."


def get_market_structure_context(symbol: str) -> dict:
    try:
        ny_tz = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        ny_tz = datetime.timezone(datetime.timedelta(hours=-5))
        
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(ny_tz)
    
    start_daily = now - datetime.timedelta(days=6)
    try:
        daily_bars = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start_daily, end=now, feed=DataFeed.IEX
        )).df
    except Exception as e:
        print(f"⚠️ Failed to fetch daily bars for {symbol}: {e}")
        daily_bars = pd.DataFrame()

    if not daily_bars.empty:
        if isinstance(daily_bars.index, pd.MultiIndex): 
            daily_bars = daily_bars.xs(symbol)
        elif 'timestamp' in daily_bars.columns:
            daily_bars = daily_bars.set_index('timestamp')
        
        daily_bars.index = pd.to_datetime(daily_bars.index)
        historical_daily = daily_bars[daily_bars.index.date < now.date()]
    else:
        historical_daily = pd.DataFrame()

    if not historical_daily.empty:
        prev_day_high = float(historical_daily['high'].iloc[-1])
        prev_day_low = float(historical_daily['low'].iloc[-1])
    else:
        prev_day_high = 0.0
        prev_day_low = 0.0
    
    start_today = datetime.datetime(now.year, now.month, now.day, 0, 0, tzinfo=ny_tz)
    try:
        five_min_bars = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame(5, TimeFrameUnit.Minute), start=start_today, end=now, feed=DataFeed.IEX
        )).df
    except Exception as e:
        print(f"⚠️ Failed to fetch intraday 5-min bars for {symbol}: {e}")
        five_min_bars = pd.DataFrame()
    
    if not five_min_bars.empty:
        if isinstance(five_min_bars.index, pd.MultiIndex): 
            five_min_bars = five_min_bars.xs(symbol)
        elif 'timestamp' in five_min_bars.columns:
            five_min_bars = five_min_bars.set_index('timestamp')
        
        five_min_bars.index = pd.to_datetime(five_min_bars.index)
        if five_min_bars.index.tz is None:
            five_min_bars.index = five_min_bars.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            five_min_bars.index = five_min_bars.index.tz_convert("America/New_York")

    if five_min_bars.empty or 'open' not in five_min_bars.columns:
        return {
            "symbol": symbol,
            "latest_close": prev_day_high if prev_day_high > 0 else 1.0, 
            "today_open": prev_day_high if prev_day_high > 0 else 1.0,
            "prev_day_high": prev_day_high,
            "prev_day_low": prev_day_low,
            "first_hour_high": prev_day_high,
            "first_hour_low": prev_day_low,
            "recent_bars_json": "[]",
            "current_time_str": now.strftime("%H:%M")
        }
    
    today_open = float(five_min_bars['open'].iloc[0])
    latest_close = float(five_min_bars['close'].iloc[-1])
    
    first_hour_bars = five_min_bars.between_time("09:30", "10:30")
    first_hour_high = float(first_hour_bars['high'].max()) if not first_hour_bars.empty else latest_close
    first_hour_low = float(first_hour_bars['low'].min()) if not first_hour_bars.empty else latest_close

    recent = five_min_bars.tail(40).copy()
    recent['time_str'] = recent.index.strftime('%H:%M')
    recent_bars_json = recent[['time_str', 'open', 'high', 'low', 'close', 'volume']].to_json(orient="records")

    return {
        "symbol": symbol,
        "latest_close": latest_close,
        "today_open": today_open,
        "prev_day_high": prev_day_high,
        "prev_day_low": prev_day_low,
        "first_hour_high": first_hour_high,
        "first_hour_low": first_hour_low,
        "recent_bars_json": recent_bars_json,
        "current_time_str": now.strftime("%H:%M")
    }