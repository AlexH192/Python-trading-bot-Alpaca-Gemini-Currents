# Python-trading-bot--Alpaca-Gemini-Currents-APIs

Program requires Python 3.10+
Program requires accounts and/or API keys from the following external services:
  * Alpaca
  * Google Gemini
  * Currents news API
  * Telegram - bot token and chat ID


To run the program:

(1) Insert API keys for Alpaca, Gemini and Currents, as well as Telegram bot token and chat ID into the empty quotes in config/settings.py.
(2) Install dependencies:
  pip install alpaca-py pandas numpy httpx pydantic google-genai yfinance apscheduler
