import json
import os
import asyncio
import datetime
import zoneinfo
from config.settings import LOG_FILE_PATH

journal_lock = asyncio.Lock()

NY_TZ = zoneinfo.ZoneInfo("America/New_York")


async def log_trade_to_journal(ticker, action, fill_price, quantity, reasoning, market_snapshot=None):
    trade_record = {
        "timestamp": datetime.datetime.now(NY_TZ).isoformat(),
        "ticker": ticker,
        "action": action,
        "fill_price": float(fill_price) if fill_price else None,
        "quantity": float(quantity) if quantity else None,
        "gemini_reasoning": reasoning,
        "market_snapshot": market_snapshot or {}
    }

    def _sync_write():
        data = []
        if os.path.exists(LOG_FILE_PATH):
            try:
                with open(LOG_FILE_PATH, "r") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
            except Exception as e:
                print(f"⚠️ Warning: Could not read trade journal: {e}")

        data.append(trade_record)

        with open(LOG_FILE_PATH, "w") as f:
            json.dump(data, f, indent=4)

    try:
        async with journal_lock:
            await asyncio.to_thread(_sync_write)
        print(f"📝 Trade successfully logged to {LOG_FILE_PATH}")
    except Exception as e:
        print(f"⚠️ Error writing trade journal: {e}")