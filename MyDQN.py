import numpy as np
import gym
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os


LR = 0.001    # learning rata

MEMORY_SIZE = 2048
BATCH_SIZE = 128

WIDTH = 50


class Net(nn.Module):

    def __init__(self, input_num, output_num):
        super(Net, self).__init__()

        self.hidden_1 = nn.Linear(input_num, WIDTH)
        self.hidden_1.weight.data.normal_(0, 0.1)

        self.hidden_2 = nn.Linear(WIDTH, WIDTH)
        self.hidden_2.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(WIDTH, output_num)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        state = self.hidden_1(state)
        state = F.sigmoid(state)
        state = self.hidden_2(state)
        state = F.sigmoid(state)
        actions = self.out(state)
        return actions


class DQN():

    def __init__(self, state_num, action_num, epsilon=0.9, gamma=0.9):

        self.state_num = state_num
        self.action_num = action_num
        self.epsilon = epsilon
        self.gamma = gamma

        self.QNet = Net(state_num, action_num)

        self.memory_count = 0
        self.memory = np.zeros((MEMORY_SIZE, state_num*2 + 2))

        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr=LR)
        # self.loss_func = nn.MSELoss()

    def loss_func(self, input, target):
        return (input - target)**2

    def get_action(self, state):

        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))

        if np.random.uniform() < self.epsilon:
            actions = self.QNet.forward(state)
            action = torch.max(actions, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, self.action_num)

        return action

    def store_transition(self, state, action, reward, state_next):
        transition = np.hstack((state, [action, reward], state_next))
        index = self.memory_count % MEMORY_SIZE
        self.memory[index, :] = transition
        self.memory_count += 1

    def learn(self):
        sample = np.random.choice(MEMORY_SIZE, BATCH_SIZE)

        sample_memory = self.memory[sample, :]
        sample_state = Variable(torch.FloatTensor(sample_memory[:, :self.state_num]))
        sample_action = Variable(torch.LongTensor(sample_memory[:, self.state_num:self.state_num+1]))
        sample_reward = Variable(torch.FloatTensor(sample_memory[:, self.state_num+1:self.state_num+2]))
        sample_state_next = sample_memory[:, -self.state_num:]

        non_terminal_mask = torch.ByteTensor(tuple(map(lambda s: not math.isnan(s[0]), sample_state_next)))
        non_terminal_state_next = Variable(torch.cat([torch.unsqueeze(torch.FloatTensor(s), 0)
                                                      for s in sample_state_next if not math.isnan(s[0])]))

        q_value = self.QNet(sample_state).gather(1, sample_action)

        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.FloatTensor))
        next_state_values[non_terminal_mask] = self.QNet(non_terminal_state_next).max(1)[0]
        next_state_values = next_state_values.view(BATCH_SIZE, 1)

        expected_q_value = sample_reward + (self.gamma * next_state_values)

        loss = self.loss_func(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward(torch.FloatTensor([1.0]))
        self.optimizer.step()

        return torch.mean(loss)


class CartPoleAgent():

    def __init__(self, epsilon=0.95, gamma=0.99, episode=1000):

        self.env = gym.make('CartPole-v0')
        self.env.seed(2000)
        self.env._max_episode_steps = 20000
        self.env.x_threshold = 2.4
        self.env.theta_threshold_radians = 0.20943951023931953

        self.epsilon = epsilon
        self.gamma = gamma
        self.episode = episode

        self.DQN = DQN(
            state_num=self.env.observation_space.shape[0],
            action_num=self.env.action_space.n,
            epsilon=self.epsilon,
            gamma=self.gamma
        )

    def get_action(self, state):
        return self.DQN.get_action(state)

    def train(self):
        plt.figure()
        x = []
        rewards = []
        avgloss = []
        for t in range(self.episode):

            state = self.env.reset()
            step = 0
            loss = 0
            while True:

                action = self.DQN.get_action(state)
                state_next, reward, done, _ = self.env.step(action)

                # modify reward
                r1 = (self.env.x_threshold - abs(state_next[0])) / self.env.x_threshold
                r2 = (self.env.theta_threshold_radians - abs(state_next[2])) / self.env.theta_threshold_radians
                reward = r1 + r2

                step += 1

                if done and step != self.env._max_episode_steps:
                    state_next = [None, None, None, None]
                    reward = -10.0

                self.DQN.store_transition(state, action, reward, state_next)

                if self.DQN.memory_count > MEMORY_SIZE:
                    loss = self.DQN.learn()

                state = state_next
                if done:
                    # print('step =', step)
                    rewards.append(step)
                    avgloss.append(loss)
                    x.append(t)
                    break
            if t == self.epsilon // 2:
                self.DQN.epsilon = 0.99
        self.saveImg(x=x, r=rewards, l=avgloss)

    def saveImg(self, x, r, l):
        plt.subplot(2, 1, 1)
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.plot(x, l)
        plt.subplot(2, 1, 2)
        plt.ylabel('reward')
        plt.xlabel('step')
        plt.plot(x, r)
        plt.savefig('CartPole.png')

    def dump(self, file='CartPoleNet0.pkl'):
        torch.save(self.DQN.QNet, file)

    def load(self, file='CartPoleNet0.pkl'):
        if os.path.exists(path=file):
            self.DQN.QNet = torch.load(file)
            return True
        else:
            return False


class MountainCarAgent():
    def __init__(self, epsilon=0.90, gamma=0.99, episode=1000):

        self.env = gym.make('MountainCar-v0')
        self.env._max_episode_steps = 2000
        self.epsilon = epsilon
        self.gamma = gamma
        self.episode = episode

        self.DQN = DQN(
            state_num=self.env.observation_space.shape[0],
            action_num=self.env.action_space.n,
            epsilon=self.epsilon,
            gamma=self.gamma
        )

    def get_action(self, state):
        return self.DQN.get_action(state)

    def train(self):
        plt.figure()
        x = []
        rewards = []
        avgloss = []
        for t in range(self.episode):

            state = self.env.reset()
            step = 0
            loss = 0
            while True:

                action = self.DQN.get_action(state)
                state_next, reward, done, _ = self.env.step(action)

                step += 1
                if done and step != self.env._max_episode_steps:
                    state_next = [None, None]

                self.DQN.store_transition(state, action, reward, state_next)

                if self.DQN.memory_count > MEMORY_SIZE:
                    loss = self.DQN.learn()

                state = state_next
                if done:
                    rewards.append(-step)
                    avgloss.append(loss)
                    x.append(t)
                    break
            if t == self.episode // 2:
                self.DQN.epsilon = 0.999
        self.saveImg(x=x, r=rewards, l=avgloss)

    def saveImg(self, x, r, l):
        plt.subplot(2, 1, 1)
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.plot(x, l)
        plt.subplot(2, 1, 2)
        plt.ylabel('reward')
        plt.xlabel('step')
        plt.plot(x, r)
        plt.savefig('MountainCar.png')

    def dump(self, file='MountainCarNet0.pkl'):
        torch.save(self.DQN.QNet, file)

    def load(self, file='MountainCarNet0.pkl'):
        if os.path.exists(path=file):
            self.DQN.QNet = torch.load(file)
            return True
        else:
            return False


class AcrobotAgent():
    def __init__(self, epsilon=0.90, gamma=0.99, episode=300):

        self.env = gym.make('Acrobot-v1')
        self.env._max_episode_steps = 2000
        self.epsilon = epsilon
        self.gamma = gamma
        self.episode = episode

        self.DQN = DQN(
            state_num=self.env.observation_space.shape[0],
            action_num=self.env.action_space.n,
            epsilon=self.epsilon,
            gamma=self.gamma
        )

    def get_action(self, state):
        return self.DQN.get_action(state)

    def train(self):
        plt.figure()
        x = []
        rewards = []
        avgloss = []
        for t in range(self.episode):

            state = self.env.reset()
            step = 0
            loss = 0
            while True:

                action = self.DQN.get_action(state)
                state_next, reward, done, _ = self.env.step(action)

                step += 1
                if done and step != self.env._max_episode_steps:
                    state_next = [None, None, None, None, None, None]

                self.DQN.store_transition(state, action, reward, state_next)

                if self.DQN.memory_count > MEMORY_SIZE:
                    loss = self.DQN.learn()

                state = state_next
                if done:
                    rewards.append(-step)
                    avgloss.append(loss)
                    x.append(t)
                    break
            if t == self.episode // 2:
                self.DQN.epsilon = 0.999
        self.saveImg(x=x, r=rewards, l=avgloss)

    def saveImg(self, x, r, l):
        plt.subplot(2, 1, 1)
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.plot(x, l)
        plt.subplot(2, 1, 2)
        plt.ylabel('reward')
        plt.xlabel('step')
        plt.plot(x, r)
        plt.savefig('Acrobot.png')

    def dump(self, file='AcrobotNet0.pkl'):
        torch.save(self.DQN.QNet, file)

    def load(self, file='AcrobotNet0.pkl'):
        if os.path.exists(path=file):
            self.DQN.QNet = torch.load(file)
            return True
        else:
            return False
