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
        self.trading_days = price_history.get_trading_days()
        self.trading_fee = trading_fee
        self.risk = risk
        self.cash = starting_balance
        self.observation_space = lookback
        self.price_history = price_history
        self.viewer = None
        self.balance = starting_balance
        self.current_step = lookback
        self.inventory = []

    def step(self, action):
        reward = self._take_action(action)

        close_price = self.trading_days[self.current_step].get_close()
        done = True if self.current_step >= len(self.price_history.get_dataframe()) or (getTotalPriceOfHoldings(self.inventory, close_price) == 0 and (self.cash * self.risk) < close_price) else False

        #plt.scatter(self.price_history.get_dataframe().iloc[self.current_step]['time'], self.balance, marker='+', label='Balance', color='blue')
        self.current_step += 1

        observation = self._get_observation(self.current_step)

        info = {}

        return observation, reward, done, info

    def reset(self):
        self.__init__(self.price_history)
        return self._get_observation(self.current_step)

    def _get_observation(self,step):
        observations = []
        for i in range(self.current_step - self.observation_space, self.current_step):
            candle = self.trading_days[i].get_candle()
            obs = [
                candle.get_direction(),
                candle.get_size(),
                candle.get_wick_down(),
                candle.get_wick_up()
            ]
            observations.append(obs)
        return observations

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            close_price = self.trading_days[self.current_step].get_close()
            print(self.trading_days[self.current_step].get_date().strftime('%m/%d/%Y') + " Cash " + formatPrice(self.cash) + " | Holdings: " + formatPriceOfHoldings(self.inventory, close_price) + " | Balance: " + formatPrice(self.balance))

    # def _start_viewer(self):
    #     mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    #     alldays = DayLocator()              # minor ticks on the days
    #     weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    #     dayFormatter = DateFormatter('%d')      # e.g., 12
    #     self.fig, self.ax = plt.subplots()
    #     self.fig.subplots_adjust(bottom=0.2)
    #     self.ax.xaxis.set_major_locator(mondays)
    #     self.ax.xaxis.set_minor_locator(alldays)
    #     self.ax.xaxis.set_major_formatter(weekFormatter)
    #
    #     candlestick_ohlc(self.ax, self.price_history.get_dataframe().values, width=0.6)
    #
    #     self.ax.xaxis_date()
    #     self.ax.autoscale_view()
    #     plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    #
    #     for i in range(len(self.price_history.get_dataframe())):
    #         y = np.random.random()
    #         plt.scatter(self.price_history.get_dataframe().iloc[i]['time'], y, marker='+', label='Balance', color='blue')
    #         plt.pause(0.05)
    #
    #     plt.show()

    def _take_action(self, action):
        close_price = self.trading_days[self.current_step].get_close()
        reward = 0

        if action == BUY and (self.cash * self.risk) > close_price + self.trading_fee:
            shares = math.floor((self.cash * self.risk) / close_price)
            cost = close_price * shares
            self.inventory.append([shares, cost])
            self.cash = self.cash - cost - self.trading_fee
            #reward -= self.trading_fee / 100 #trading fee
            print("Buy " + str(shares) + " shares @ " + formatPrice(close_price))

        elif action == SELL and len(self.inventory) > 0:  # sell
            order = self.inventory.pop(0)
            bought_price = order[1]
            bought_shares = order[0]
            current_value = bought_shares * close_price
            self.cash += current_value - self.trading_fee
            take = current_value - bought_price
            reward = 2 if take - self.trading_fee > 0 else max(take, -1) - (self.trading_fee / 100)
            print("Sell " + str(bought_shares) + " @ " + formatPrice(close_price) + " [Net " + formatPrice(current_value - bought_price) + "]")

        self.balance = getTotalPriceOfHoldings(self.inventory, close_price) + self.cash
        return reward
