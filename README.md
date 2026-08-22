# Python-trading-bot--Alpaca-Gemini-Currents-APIs

Program requires Python 3.10+
Program requires accounts and/or API keys from the following external services:
  * Alpaca
  * Google Gemini
  * Currents news API
  * Telegram - bot token and chat ID

Certain file paths are hardcoded -- need to be replaced manually: bot_swing_commodities.py and enrich_trade_journal.py


To run the program:

(1) Insert API keys for Alpaca, Gemini and Currents, as well as Telegram bot token and chat ID into the empty quotes in config/settings.py.
(2) Install dependencies:
  pip install alpaca-py pandas numpy httpx pydantic google-genai yfinance apscheduler
(3) Run command in terminal/command prompt/console:

  [On a mac laptop, to keep program awake even when screen turns off:
  [ caffeinate -i python3 'FILEPATH/bot_intraday_etfs.py'
  [ caffeinate -i python3 'FILEPATH/bot_swing_commodities.py'
   
  For other devices:
   python3 'FILEPATH/bot_intraday_etfs.py'
   python3 'FILEPATH/bot_swing_commodities.py'


To enrich trade journal based on historical market data pulled separately, run:
python3 'FILEPATH/Logs/enrich_trade_journal.py' --reconciled 'FILEPATH/Logs/reconciled_trades.json' --out enriched_trades.json
