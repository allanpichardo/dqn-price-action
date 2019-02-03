import matplotlib.dates as dates
import pandas as pd


class PriceHistory:

    def __init__(self, path):
        self._raw_df = pd.read_csv(path, index_col='Date', parse_dates=True, infer_datetime_format=True)
        self._raw_df.insert(loc=0, column='time', value=dates.date2num(self._raw_df.index.values))

    def get_dataframe(self):
        return self._raw_df

    def get_trading_days(self):
        days = []
        for index, row in self._raw_df.iterrows():
            days.append(TradingDay(row))

        return days


class TradingDay:

    def __init__(self, raw_day):
        self._open = raw_day['Open']
        self._high = raw_day['High']
        self._low = raw_day['Low']
        self._close = raw_day['Close']
        self._adj_close = raw_day['Adj Close']
        self._volume = raw_day['Volume']
        self._date = raw_day.name

    def get_open(self):
        return self._open

    def get_high(self):
        return self._high

    def get_low(self):
        return self._low

    def get_close(self):
        return self._close

    def get_adjusted_close(self):
        return self._adj_close

    def get_volume(self):
        return self._volume

    def get_date(self):
        return self._date

    def get_candle(self):
        return Candle(self)


class Candle:

    def __init__(self, trading_day):
        self._direction = 1 if trading_day.get_close() > trading_day.get_open() else -1
        self._direction = 0 if trading_day.get_close() == trading_day.get_open() else self._direction
        self._wick_up = (trading_day.get_high() - trading_day.get_close()) / trading_day.get_close()
        self._wick_down = (trading_day.get_low() - trading_day.get_open()) / trading_day.get_open()
        self._size = abs((trading_day.get_close() - trading_day.get_open()) / trading_day.get_open())

    def get_direction(self):
        return self._direction

    def get_wick_up(self):
        return self._wick_up

    def get_wick_down(self):
        return self._wick_down

    def get_size(self):
        return self._wick_down
