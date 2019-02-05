import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

# Get the environment and extract the number of actions.
from envs.yahoofinance_env import YahooFinanceEnv
from ml.yahoofinance import PriceHistory
from rl.agents import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

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
EPISODES = 100000
INTERVAL = price_history.get_length()
STEPS = INTERVAL * EPISODES
MEMORY_WINDOW_LENGTH = 5

print(env.get_shape())

model = Sequential()
model.add(Flatten(input_shape=(MEMORY_WINDOW_LENGTH,) + env.get_shape()))
model.add(Dense(6))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('tanh'))
model.add(Dense(3))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(1000000, window_length=MEMORY_WINDOW_LENGTH)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=INTERVAL)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=INTERVAL, gamma=.99, target_model_update=INTERVAL,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We capture the interrupt exception so that training
# can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
weights_filename = 'save/dqn_{}_weights.h5f'.format('act')
checkpoint_weights_filename = 'save/dqn_weights_{step}.h5f'
log_filename = 'save/dqn_{}_log.json'.format('act')
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=INTERVAL)]
dqn.fit(env, nb_steps=STEPS, log_interval=INTERVAL, visualize=False, callbacks=callbacks)

# After training is done, we save the final weights one more time.
dqn.save_weights(weights_filename, overwrite=True)
# Finally, evaluate our algorithm for 10 episodes.
dqn.test(env, nb_episodes=10, visualize=True)

# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
# dqn.fit(env, nb_steps=len(price_history.get_dataframe()) - 5, visualize=True, verbose=2, callbacks=[TrainEpisodeLogger()], )

# # After training is done, we save the final weights.
# dqn.save_weights('dqn_{}_weights.h5f'.format('price_action'), overwrite=True)

# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)
