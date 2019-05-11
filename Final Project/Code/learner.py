# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import warnings
# from environment import Environment
from environment_1 import Environment



EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory_size = 1000
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

#        if len(self.memory) > self.memory_size:
            # self.memory = self.memory[-self.memory_size:]

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    env = Environment()
    state_size = env.state_space_n
    action_size = env.action_space_n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    f= open("model.txt","w+")

    for e in range(EPISODES):
        f.write("Episode: " + str(e) + "\n")
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, congestion, done = env.step(action)
            reward = reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            print('\nepisode: ', e,'time step: ', time, "reward: ", reward, "state: ", state)
            f.write('episode: '+ str(e) +'time step: ' +  str(time) + ", reward: " + str(reward) + ", prices: " + str(state) + ", congestion: " + str(congestion) + '\n')
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, reward, agent.epsilon))
                f.write("episode: {}/{}, score: {}, e: {:.2} \n"
                      .format(e, EPISODES, reward, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

    f.close()
    env.close_pool()
