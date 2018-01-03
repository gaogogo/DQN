import gym
import numpy as np
import sys
import getopt


import MyQLearning
import MyDQN
import MyImprovedDQN


def helpInfo():
    print('python/python3 testAcrobot.py -h\t get the help information')
    print('python/python3 testAcrobot.py -q -Q\t test the Acrobot with q-learning')
    print('python/python3 testAcrobot.py -d -D\t test the Acrobot with DQN')
    print('python/python3 testAcrobot.py -i -I\t test the Acrobot with improved DQN')


def testQlearning():
    agent = MyQLearning.AcrobotAgent()

    if not agent.load():
        agent.train()
        agent.dump()

    env = gym.make('Acrobot-v1')
    env._max_episode_steps = 2000
    agent.epsilon = 1.0

    rewards = []

    for test in range(100):
        obs = env.reset()
        reward = 0
        while True:
            #env.render()
            state = agent.get_state(obs)
            action = agent.get_action(state)
            obs, r, done, _ = env.step(action)
            reward += r
            if done:
                rewards.append(reward)
                print('reward =', reward)
                break

    mean = np.mean(rewards)
    std = np.std(rewards)
    print('mean =', mean)
    print('std =', std)


def testDQN():
    agent = MyDQN.AcrobotAgent()

    if not agent.load():
        agent.train()
        agent.dump()

    env = gym.make('Acrobot-v1')
    env._max_episode_steps = 2000
    agent.DQN.epsilon = 1.0

    rewards = []

    for test in range(100):
        state = env.reset()
        reward = 0
        while True:
            #env.render()
            action = agent.get_action(state)
            state, r, done, _ = env.step(action)
            reward += r
            if done:
                rewards.append(reward)
                print('reward =', reward)
                break

    mean = np.mean(rewards)
    std = np.std(rewards)
    print('mean =', mean)
    print('std =', std)


def testImproveDQN():
    agent = MyImprovedDQN.AcrobotAgent()

    if not agent.load():
        agent.train()
        agent.dump()

    env = gym.make('Acrobot-v1')
    env._max_episode_steps = 2000
    agent.DQN.epsilon = 1.0

    rewards = []

    for test in range(100):
        state = env.reset()
        reward = 0
        while True:
            #env.render()
            action = agent.get_action(state)
            state, r, done, _ = env.step(action)
            reward += r
            if done:
                rewards.append(reward)
                print('reward =', reward)
                break

    mean = np.mean(rewards)
    std = np.std(rewards)
    print('mean =', mean)
    print('std =', std)


def main(argv):

    try:
        opts, args = getopt.getopt(argv, 'hqQdDiI')
    except getopt.GetoptError:
        helpInfo()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            helpInfo()
            sys.exit()
        elif opt in ('-q', '-Q'):
            testQlearning()
            sys.exit()
        elif opt in ('-d', '-D'):
            testDQN()
            sys.exit()
        elif opt in ('-i', '-I'):
            testImproveDQN()
            sys.exit()

    helpInfo()


if __name__ == "__main__":
   main(sys.argv[1:])