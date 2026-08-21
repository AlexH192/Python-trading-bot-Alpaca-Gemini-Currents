from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

# Initialize clients globally
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trade_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True) 
stream_client = TradingStream(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)