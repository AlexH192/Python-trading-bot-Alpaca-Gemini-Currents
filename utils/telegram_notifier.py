import datetime
import zoneinfo
import httpx
from config.settings import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

class TelegramNotifier:
    """Dispatches clean markdown formatting blocks directly to your personal Telegram thread."""
    @staticmethod
    def get_timestamp_prefix() -> str:
        try:
            ny_tz = zoneinfo.ZoneInfo("America/New_York")
        except Exception:
            ny_tz = datetime.timezone(datetime.timedelta(hours=-5))
        now = datetime.datetime.now(ny_tz)
        return f"[{now.strftime('%Y-%m-%d %H:%M:%S')} EST]\n"

    @staticmethod
    async def send(message: str):
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        full_message = TelegramNotifier.get_timestamp_prefix() + message
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": full_message, "parse_mode": "HTML"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    print(f"Telegram API response anomaly: {response.text}")
        except Exception as e:
            print(f"Telegram notification failed: {e}")