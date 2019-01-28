from ml.agent import Agent
from functions import *
import sys

if len(sys.argv) != 5:
    print("Usage: python train.py [stock] [window] [episodes] [starting balance]")
    exit()

stock_name, window_size, episode_count, starting_balance = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

trading_fee = 5
risk = 0.1
agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
    cash = starting_balance
    print("Episode " + str(e) + "/" + str(episode_count))
    print("Balance: $" + str(cash))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(state)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 0:
            print("Cash " + formatPrice(cash) + " | Holdings: " + formatPriceOfHoldings(agent.inventory, data[t][3]))
        if action == 1 and (cash * risk) > data[t][3]:  # buy
            shares = math.floor((cash * risk) / data[t][3])
            cost = data[t][3] * shares
            agent.inventory.append([shares, cost])
            cash = cash - cost - trading_fee
            reward -= trading_fee #trading fee
            print("Buy " + str(shares) + " shares @ " + formatPrice(data[t][3]) + " | Balance: " + formatPrice(cash))

        elif action == 2 and len(agent.inventory) > 0:  # sell
            order = agent.inventory.pop(0)
            bought_price = order[1]
            bought_shares = order[0]
            current_value = bought_shares * data[t][3]
            cash += current_value
            reward = max(current_value - bought_price, 0) - trading_fee
            #reward = (current_value - bought_price) - trading_fee #subtracting trading fee
            cash -= trading_fee
            total_profit += current_value - bought_price
            print("Sell " + str(bought_shares) + " @ " + formatPrice(data[t][3]) + " | Profit: " + formatPrice(current_value - bought_price))

        done = True if (t == l - 1) or (getTotalPriceOfHoldings(agent.inventory, data[t][3]) == 0 and (cash * risk) < data[t][3]) else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        account = getTotalPriceOfHoldings(agent.inventory, data[t][3]) + cash
        print("Account: "+formatPrice(account))

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(account - starting_balance) + " " + str(((account-starting_balance)/starting_balance)*100) + "%")
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if done:
            break

    if e % 10 == 0:
        agent.model.save("models/model_ep" + str(e))
