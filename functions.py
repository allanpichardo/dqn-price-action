import numpy as np
import math


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def formatPriceOfHoldings(holdings, current_price):
    return formatPrice(getTotalPriceOfHoldings(holdings, current_price))


def getTotalPriceOfHoldings(holdings, current_price):
    total = 0
    for h in holdings:
        total += h[0] * current_price
    return total

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        temp = []
        linarr = np.array(line.split(","))
        for j in range(1, 5):
            temp.append(float(linarr[j]))

        vec.append(temp)

    return vec


# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# returns an an n-day state representation ending at time t
def getState(data, t, n, close_col=3):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
    res = []
    for i in range(n - 1):
        temp = []
        for j in range(4):
            temp.append(sigmoid(block[i + 1][j] - block[i][j]))
        res.append(temp)
        #res.append(sigmoid(block[i + 1][close_col] - block[i][close_col]))

    return np.array([res])
