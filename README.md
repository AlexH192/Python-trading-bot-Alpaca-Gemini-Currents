# Python-trading-bot--Alpaca-Gemini-Currents-APIs
This is a Git repo for an algorithmic trading bot tracking (1) stock futures and (2) commodity futures, executing trades on their respective ETFs (e.g. QQQ, SPY, SOXX), running locally on Python and pulling data, executing trades and sending notifications via external API services.
<br><br> The equities (stock futures) trading strategy is twofold: at market open, an Opening Range Breakout (ORB) strategy is executed, subsequently a liquidity sweep strategy targets larger moves later into the trading session.
<br><br> The commodities (commodity futures) trading strategy is a mean-reversion swing-trading strategy relying on longer timeframes and holding positions for multiple days.
<br><br> Trades are executed via the Alpaca API. Alpaca is a platform where paper-trading (trading with simulated money) is available. Trading with real money is also possible, however testing is done strictly with simulated money.
<br><br>Some configurations have been made to this repo in order to make it compatible with the Kubernetes cluster it will run on. The cluster's GitHub repo is linked <a href="https://github.com/AlexH192/k3s-cluster.git">here.</a>
## General
Program requires Python 3.10+, as well as accounts and/or API keys from the following external services:
  * Alpaca
  * Google Gemini
  * Currents News API
  * Telegram - bot token and chat ID

## Setup & Running the Program
To run the program locally:

(1) Insert all needed API keys or credentials (Alpaca, Gemini, Currents, Telegram, Redis) into the .env file, as per the template `.env_example`.
<br>(2) Install dependencies as listed in `requirements.txt`:
  ```
alpaca-py
requests
pandas
numpy
pandas-ta-classic
apscheduler
pydantic
google-genai
httpx
python-dotenv
tzdata
redis
yfinance
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
This data may be fed back into Gemini in order to inform decisions based on past trade outcomes in future versions.
