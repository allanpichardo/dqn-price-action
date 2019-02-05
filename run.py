import numpy as np
import gym

import keras.backend as K
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Conv1D, LSTM
from keras.optimizers import Adam

# Get the environment and extract the number of actions.
from envs.yahoofinance_env import YahooFinanceEnv
from rl.agents import DQNAgent
from rl.callbacks import CallbackList, TestLogger, TrainEpisodeLogger
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from ml.yahoofinance import PriceHistory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

price_history = PriceHistory('data/AMD.csv')

env = YahooFinanceEnv(price_history, risk=0.25, lookback=1)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space

#input_shape = (5,) + (1, 4)
input_shape = (1,)
batch_size = 1
timesteps = 5
parameters = 4

print(env.get_shape())

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.get_shape()))
model.add(Dense(1))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(1000000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=1000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We capture the interrupt exception so that training
# can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
weights_filename = 'save/dqn_{}_weights.h5f'.format('act')
checkpoint_weights_filename = 'save/dqn_weights_{step}.h5f'
log_filename = 'save/dqn_{}_log.json'.format('act')
# callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
# callbacks += [FileLogger(log_filename, interval=100)]
for i in range(50000):
    print("Epoch {} start".format(i))
    dqn.fit(env, nb_steps=len(price_history.get_dataframe()), log_interval=10000, visualize=False)
    print("Epoch {} Finished".format(i))

# After training is done, we save the final weights one more time.
dqn.save_weights(weights_filename, overwrite=True)
# Finally, evaluate our algorithm for 10 episodes.
dqn.test(env, nb_episodes=10, visualize=False)

# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
# dqn.fit(env, nb_steps=len(price_history.get_dataframe()) - 5, visualize=True, verbose=2, callbacks=[TrainEpisodeLogger()], )

# # After training is done, we save the final weights.
# dqn.save_weights('dqn_{}_weights.h5f'.format('price_action'), overwrite=True)

# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)
