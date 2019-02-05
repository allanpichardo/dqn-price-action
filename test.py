from ml.yahoofinance import *

ph = PriceHistory('data/AMD.csv')
print(ph.get_trading_days()[0].get_candle().to_vector())