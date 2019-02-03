import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

# Get the environment and extract the number of actions.
from envs.yahoofinance_env import YahooFinanceEnv
from rl.agents import DQNAgent
from rl.callbacks import CallbackList, TestLogger, TrainEpisodeLogger
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from ml.yahoofinance import PriceHistory

price_history = PriceHistory('data/AMD.csv')

env = YahooFinanceEnv(price_history, risk=0.25)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + (5,4)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, gamma=0.99)
dqn.compile(Adam(lr=1e-4), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=len(price_history.get_dataframe()) - 5, visualize=True, verbose=2, callbacks=[TrainEpisodeLogger()])

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format('price_action'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=50000, visualize=True)
