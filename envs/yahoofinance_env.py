import math
import os
import signal

import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY

from envs.actions import *
from envs.mpl_finance import candlestick_ohlc
from functions import formatPrice, getTotalPriceOfHoldings, formatPriceOfHoldings
from ml.yahoofinance import PriceHistory


class YahooFinanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    action_space = 3

    def __init__(self, price_history: PriceHistory, starting_balance=1000, lookback=5, trading_fee=4.99, risk=0.1):
        self._starting_balance = starting_balance
        self._lookbabck = lookback
        self.trading_days = price_history.get_trading_days()
        self.trading_fee = trading_fee
        self.risk = risk
        self.cash = starting_balance
        self.observation_space = 4
        self.price_history = price_history
        self.viewer = None
        self.balance = starting_balance
        self.current_step = lookback
        self.inventory = []

    def step(self, action):
        if self.current_step >= self.price_history.get_length():
            return [[0,0,0,0,0,0]], 0, True, {}

        reward = self._take_action(action)

        close_price = self.trading_days[self.current_step].get_close()
        done = True if self.current_step >= len(self.price_history.get_dataframe()) or (getTotalPriceOfHoldings(self.inventory, close_price) == 0 and (self.cash * self.risk) < close_price) else False

        self.current_step += 1

        observation = self._get_observation(self.current_step)

        info = {}

        return observation, reward, done, info

    def get_shape(self):
        input = self._get_observation(self.current_step)
        return input.shape


    def reset(self):
        self.__init__(self.price_history, starting_balance=self._starting_balance, lookback=self._lookbabck, trading_fee=self.trading_fee, risk=self.risk)
        return self._get_observation(self.current_step)

    def _get_observation(self,step):
        observations = []

        for i in range(self.current_step - self._lookbabck, self.current_step):
            candle = self.trading_days[i].get_candle()
            observations.append(candle.to_vector())

        observations = np.array(observations)
        return observations

    def render(self, mode='human', close=False):
        if self.current_step >= self.price_history.get_length():
            return

        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            close_price = self.trading_days[self.current_step].get_close()
            print(self.trading_days[self.current_step].get_date().strftime('%m/%d/%Y') + " Cash " + formatPrice(self.cash) + " | Holdings: " + formatPriceOfHoldings(self.inventory, close_price) + " | Balance: " + formatPrice(self.balance))

    def _take_action(self, action):
        close_price = self.trading_days[self.current_step].get_close()
        reward = 0

        if action == BUY and (self.cash * self.risk) > close_price + self.trading_fee:
            shares = math.floor((self.cash * self.risk) / close_price)
            cost = close_price * shares
            self.inventory.append([shares, cost])
            self.cash = self.cash - cost - self.trading_fee
            #print("Buy " + str(shares) + " shares @ " + formatPrice(close_price))

        elif action == SELL and len(self.inventory) > 0:  # sell
            order = self.inventory.pop(0)
            bought_price = order[1]
            bought_shares = order[0]
            current_value = bought_shares * close_price
            self.cash += current_value - self.trading_fee
            take = (current_value - bought_price)
            reward = math.tanh(take)
            #print("Sell " + str(bought_shares) + " @ " + formatPrice(close_price) + " [Net " + formatPrice(current_value - bought_price) + "]")

        self.balance = getTotalPriceOfHoldings(self.inventory, close_price) + self.cash
        return reward
