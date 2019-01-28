import random
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam


class Agent:

    def __init__(self, state_size, action_size=3, learning_rate=0.001, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = action_size
        self.is_eval = is_eval
        self.learning_rate = learning_rate
        self.memory = []

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model("models/" + model_name) if is_eval else self._model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)
        # Extract informations from each memory
        for state, action, reward, next_state, done in minibatch:
            # if done, make our target reward
            target = reward
            if not done:
                # predict the future discounted reward
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that target_f
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Train the Neural Net with the state and target_f
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def _model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.state_size, 4)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='relu'))

        model.compile(Adam(lr=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0])
