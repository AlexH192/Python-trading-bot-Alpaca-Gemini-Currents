# Python-trading-bot--Alpaca-Gemini-Currents-APIs
This is a Git repo for an algorithmic trading bot tracking (1) stock futures and (2) commodity futures, executing trades on their respective ETFs (e.g. QQQ, SPY, SOXX), running locally on Python and pulling data, executing trades and sending notifications via external API services.
<br><br> The equities (stock futures) trading strategy is twofold: at market open, an Opening Range Breakout (ORB) strategy is executed, subsequently a liquidity sweep strategy targets larger moves later into the trading session.
<br><br> The commodities (commodity futures) trading strategy is a mean-reversion swing-trading strategy relying on longer timeframes and holding positions for multiple days.
<br><br> Trades are executed via the Alpaca API. Alpaca is a platform where paper-trading (trading with simulated money) is available. Trading with real money is also possible, however testing is done strictly with simulated money.

## General
Program requires Python 3.10+
Program requires accounts and/or API keys from the following external services:
  * Alpaca
  * Google Gemini
  * Currents news API
  * Telegram - bot token and chat ID

## Setup & Running the Program
To run the program:

(1) Insert API keys for Alpaca, Gemini and Currents, as well as Telegram bot token and chat ID into the empty quotes in config/settings.py.
<br>(2) Install dependencies:
  ```
pip install alpaca-py pandas numpy httpx pydantic google-genai yfinance apscheduler
```
<br>(3) Run command in terminal/command prompt/console:

  <br>On a mac laptop, to keep program awake even when screen turns off:
  ```
  caffeinate -i python3 'FILEPATH/bot_intraday_etfs.py'
  caffeinate -i python3 'FILEPATH/bot_swing_commodities.py'
   ``````
  For other devices:
  ```
   python3 'FILEPATH/bot_intraday_etfs.py'
   python3 'FILEPATH/bot_swing_commodities.py'
  ```


To enrich trade journal based on historical market data pulled separately, run:
```
python3 'FILEPATH/Logs/enrich_trade_journal.py' --reconciled 'FILEPATH/Logs/reconciled_trades.json' --out enriched_trades.json
```
