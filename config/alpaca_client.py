from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

# Initialize shared clients
trade_client = TradingClient(
    api_key=ALPACA_API_KEY, 
    secret_key=ALPACA_SECRET_KEY, 
    paper=ALPACA_PAPER
)

data_client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY, 
    secret_key=ALPACA_SECRET_KEY
)