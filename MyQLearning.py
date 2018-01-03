import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os


class CartPoleAgent():
    """q-learning agent"""
    def __init__(self, alpha=0.001, gamma=0.99, epsilon=0.95, episode=5000):
        "get the env info"
        self.env = gym.make('CartPole-v0')
        self.env._max_episode_steps = 20000

        self.action_size = self.env.action_space.n
        self.actions = np.arange(self.action_size)

        "set q-learning parameter"
        self.alpha = alpha     # learning rate
        self.gamma = gamma    # attenuation rate
        self.epsilon = epsilon  # epsilon greedy
        self.episode = episode  # trian time

        self.QTable = {}

        self.CART_POS = np.linspace(-2.4, 2.4, 24)
        self.CART_VEL = np.linspace(-5.0, 5.0, 5)
        self.POLE_ANG = np.linspace(-0.2, 0.2, 10)
        self.POLE_VEL = np.linspace(-5.0, 5.0, 5)

        self.pos_threshold = 2.4
        self.ang_threshold = 0.20943951023931953

    def get_state(self, observation):
        "trans obs to state"
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        x = np.digitize(x=[cart_position], bins=self.CART_POS)[0]
        x_vel = np.digitize(x=[cart_velocity], bins=self.CART_VEL)[0]
        ang = np.digitize(x=[pole_angle], bins=self.POLE_ANG)[0]
        ang_vel = np.digitize(x=[pole_velocity], bins=self.POLE_VEL)[0]
        return (x, x_vel, ang, ang_vel)

    def get_action(self, state):

        if np.random.uniform() < self.epsilon:
            # choose best action
            x = []
            count = 0
            for action in self.actions:
                if (state, action) not in self.QTable:
                    self.QTable[(state, action)] = 0
                    count += 1
                x.append(self.QTable[(state, action)])
            if count == self.action_size:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(x)
        else:
            # choose random action
            action = self.env.action_space.sample()
        return action

    def train(self):
        plt.figure()
        x = []
        y = []
        # stepcount = 0
        for t in range(self.episode):

            observation = self.env.reset()

            step = 0
            while True:
                # env.render()

                state = self.get_state(observation)
                action = self.get_action(state)
                if (state, action) not in self.QTable:
                    self.QTable[(state, action)] = 0

                observation, reward, done, _ = self.env.step(action)

                state_predict = self.get_state(observation)
                action_predict = self.get_action(state_predict)
                if (state_predict, action_predict) not in self.QTable:
                    self.QTable[(state_predict, action_predict)] = 0

                r1 = self.pos_threshold - abs(observation[0]) / self.pos_threshold
                r2 = self.ang_threshold - abs(observation[2]) / self.ang_threshold
                if r1 < 0:
                    r1 = -100
                if r2 < 0:
                    r2 = -100

                reward = r1 + r2

                "update QTable"
                self.QTable[(state, action)] = self.QTable[(state, action)] + self.alpha * \
                                               (reward + self.gamma*self.QTable[(state_predict, action_predict)])
                step += 1
                if done:
                    x.append(t)
                    y.append(step)
                    plt.clf()
                    plt.xlabel('episode')
                    plt.ylabel('step')
                    plt.plot(x, y)
                    plt.pause(0.001)
                    break
            if t == self.episode // 2:
                self.epsilon = 0.999

        plt.ioff()
        plt.show()

    def dump(self, path='CartPoleQTable.pkl'):
        file = open(path, 'wb')
        pkl.dump(obj=self.QTable, file=file)

    def load(self, path='CartPoleQTable.pkl'):
        if os.path.exists(path=path):
            file = open(path, 'rb')
            self.QTable = pkl.load(file=file)
            return True
        else:
            return False


class MountainCarAgent():
    def __init__(self, alpha=0.006, gamma=0.99, epsilon=0.90, episode=700):
        "get the env info"
        self.env = gym.make('MountainCar-v0')
        self.env._max_episode_steps = 2000
        self.action_size = self.env.action_space.n
        self.actions = np.arange(self.action_size)

        "set q-learning parameter"
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # attenuation rate
        self.epsilon = epsilon  # epsilon greedy

        self.episode = episode  # trian times
        self.QTable = {}

        self.CAR_POS = np.linspace(-1.2, 0.6, 16)
        self.CAR_VEL = np.linspace(-0.07, 0.07, 16)

    def get_state(self, observation):
        car_position, car_velocity = observation
        pos = np.digitize(x=[car_position], bins=self.CAR_POS)[0]
        vel = np.digitize(x=[car_velocity], bins=self.CAR_VEL)[0]
        return (pos, vel)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            # choose best action
            x = []
            count = 0
            for action in self.actions:
                if (state, action) not in self.QTable:
                    self.QTable[(state, action)] = 0
                    count += 1
                x.append(self.QTable[(state, action)])
            if count == self.action_size:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(x)
        else:
            # choose random action
            action = self.env.action_space.sample()
        return action

    def train(self):
        plt.figure()
        x = []
        y = []
        for t in range(self.episode):

            observation = self.env.reset()

            r = 0
            while True:
                # env.render()

                state = self.get_state(observation)
                action = self.get_action(state)
                if (state, action) not in self.QTable:
                    self.QTable[(state, action)] = 0

                observation, reward, done, _ = self.env.step(action)

                state_predict = self.get_state(observation)
                action_predict = self.get_action(state_predict)
                if (state_predict, action_predict) not in self.QTable:
                    self.QTable[(state_predict, action_predict)] = 0

                "update QTable"
                self.QTable[(state, action)] = self.QTable[(state, action)] + self.alpha * \
                                               (reward + self.gamma*self.QTable[(state_predict, action_predict)])
                r += reward
                if done:
                    x.append(t)
                    y.append(r)
                    plt.clf()
                    plt.xlabel('episode')
                    plt.ylabel('reward')
                    plt.plot(x, y)
                    plt.pause(0.001)
                    break
        plt.ioff()
        plt.show()

    def dump(self, path='MountainCarQTable.pkl'):
        file = open(path, 'wb')
        pkl.dump(obj=self.QTable, file=file)

    def load(self, path='MountainCarQTable.pkl'):
        if os.path.exists(path=path):
            file = open(path, 'rb')
            self.QTable = pkl.load(file=file)
            return True
        else:
            return False


class AcrobotAgent():
    def __init__(self, alpha=0.001, gamma=0.99, epsilon=0.90, episode=2000):
        "get the env info"
        self.env = gym.make('Acrobot-v1')
        self.env._max_episode_steps = 2000

        self.action_size = self.env.action_space.n
        self.actions = np.arange(self.action_size)

        "set q-learning parameter"
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # attenuation rate
        self.epsilon = epsilon  # epsilon greedy

        self.episode = episode  # trian times
        self.QTable = {}

        self.ARC_THETA = np.linspace(-np.pi, np.pi, 24)
        self.ARC_VEL0 = np.linspace(-4*np.pi, 4*np.pi, 8)
        self.ARC_VEL1 = np.linspace(-9*np.pi, 9*np.pi, 8)

        self.env.seed(1000)

    def get_state(self, observation):
        cos0, sin0, cos1, sin1, vel0, vel1 = observation

        if sin0 >= 0:
            theta1 = np.arccos(cos0)
        else:
            theta1 = -np.arccos(cos0)

        if sin1 >= 0:
            theta2 = np.arccos(cos1)
        else:
            theta2 = -np.arccos(cos1)

        theta = np.digitize(x=[theta1], bins=self.ARC_THETA)[0]
        theta_ = np.digitize(x=[theta2], bins=self.ARC_THETA)[0]
        vel = np.digitize(x=[vel0], bins=self.ARC_VEL0)[0]
        vel_ = np.digitize(x=[vel1], bins=self.ARC_VEL1)[0]
        return (theta, theta_, vel, vel_)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            # choose best action
            x = []
            count = 0
            for action in self.actions:
                if (state, action) not in self.QTable:
                    self.QTable[(state, action)] = 0
                    count += 1
                x.append(self.QTable[(state, action)])
            if count == self.action_size:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(x)
        else:
            # choose random action
            action = self.env.action_space.sample()
        return action

    def train(self):
        plt.figure()
        x = []
        y = []
        for t in range(self.episode):

            observation = self.env.reset()

            step = 0
            #self.epsilon = 1.0 - 0.5*(0.99**t)
            while True:
                state = self.get_state(observation)
                action = self.get_action(state)
                if (state, action) not in self.QTable:
                    self.QTable[(state, action)] = 0

                observation, reward, done, _ = self.env.step(action)

                state_predict = self.get_state(observation)
                action_predict = self.get_action(state_predict)
                if (state_predict, action_predict) not in self.QTable:
                    self.QTable[(state_predict, action_predict)] = 0

                step += 1
                if done and step != self.env._max_episode_steps:
                    reward = 500

                "update QTable"
                self.QTable[(state, action)] = self.QTable[(state, action)] + self.alpha * \
                                               (reward + self.gamma*self.QTable[(state_predict, action_predict)])

                if done:
                    x.append(t)
                    y.append(step)
                    plt.clf()
                    plt.xlabel('episode')
                    plt.ylabel('step')
                    plt.plot(x, y)
                    plt.pause(0.001)
                    break
        plt.ioff()
        plt.show()

    def dump(self, path='AcrobotQTable.pkl'):
        file = open(path, 'wb')
        pkl.dump(obj=self.QTable, file=file)

    def load(self, path='AcrobotQTable.pkl'):
        if os.path.exists(path=path):
            file = open(path, 'rb')
            self.QTable = pkl.load(file=file)
            return True
        else:
            return False