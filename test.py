from ml.yahoofinance import *

ph = PriceHistory('data/AMD.csv')
print(ph.get_dataframe().head())