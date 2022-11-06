import numpy as np
import torch
import gym
from gym import spaces
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor, LongTensor
from collections import namedtuple

# Cross Entropy Method

# Wrapper to transform observation from discrete to list


class Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(Wrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        shape = (env.observation_space.n, )

        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res

# Defining the architecture of the agent


class Agent(nn.Module):
    def __init__(self, obs_size, act_size, hidden_layer):
        super(Agent, self).__init__()

        # neural net architecture
        self.neuralnet = nn.Sequential(nn.Linear(
            obs_size, hidden_layer), nn.ReLU(), nn.Linear(hidden_layer, act_size))

    # Feedforward
    def forward(self, obs):
        return (self.neuralnet(obs))


# Stores results of steps in a tuple
# stores results of each step
step_arr = namedtuple("Steps", field_names=['obs', 'act'])
# stores the rewards and steps
episode_arr = namedtuple("Episode", field_names=['reward', 'steps'])


def entropyMethod(model, episode=max_ep, batch_size=max_step, discount=gamma, alpha=alpha, cliff=False, sto=False):

    # average array
    arr = []

    env = Wrapper(model)

    # getting the sizes of observation and action space
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    # setting the neural network
    net = Agent(obs_size, act_size, hidden_layer)

    crossEntropy = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params=net.parameters(), lr=alpha)

    # Defining the batch
    def batchSet(env, NN, batch_size):
        # initial parameters
        batch = []
        episode_rewards = 0.0
        episode_steps = []
        obs = env.reset()
        softmax = nn.Softmax(dim=1)

        while True:

            # convert numpy array to tensor
            obs_tensor = torch.FloatTensor([obs])

            # getting the probabilities from initial observations
            act_probability = softmax(NN(obs_tensor))

            # get the probabilities only since [1] is the dataype
            act_prob = act_probability.data.numpy()[0]

            a = np.random.choice(len(act_prob), p=act_prob)

            # get next state and reward from action
            new_obs, reward, done, _ = env.step(a)

            episode_rewards += reward  # cumulative rewards

            # storing results of steps
            episode_steps.append(step_arr(obs=obs, act=a))

            if done:

                # storing results of episodes
                batch.append(episode_arr(
                    reward=episode_rewards, steps=episode_steps))
                episode_steps = []  # resetting if batch is not equal to desired size
                episode_rewards = 0.0

                new_obs = env.reset()

                if len(batch) == batch_size:

                    yield batch

                    batch = []

            obs = new_obs

    # storing summation results
    sum_arr = []
    total = 0.0
    counter = 0

    def bestBatch(batch, discount=gamma, percentile=percent):

        train_obs = []  # stores the best observations based on rewards
        train_act = []  # stores the best actions based on rewards
        training = []  # a combined list

        rewardList = list(map(lambda x: x.reward, batch)
                          )  # map the rewards list
        rewardBest = list(
            map(lambda s: s.reward*(discount**len(s.steps)), batch))
        cliffBound = np.percentile(rewardList, percentile)
        rewardBound = np.percentile(rewardBest, percentile)  # get the bound
        rewardMean = float(np.mean(rewardList))
        rewardSum = float(np.sum(rewardList))

        for reward, example in zip(rewardBest, batch):
            if reward > rewardBound:  # for batches with rewards greater than bound, use for training
                train_obs.extend(map(lambda step: step.obs, example.steps))
                train_act.extend(map(lambda step: step.act, example.steps))
                training.append(example)
            else:
                None

        return (train_obs, train_act, training, rewardMean, rewardSum)

    for i, batch in enumerate(batchSet(env, net, batch_size)):

        train, labels, data, mean, rewardSum = bestBatch(
            batch, percentile=percent)

        # storing the average rewards
        arr = np.append(arr, rewardSum)
        if i % 40 == 0:
            average = np.average(arr[i-40:i])
            new_count = i/40

        sum_arr = np.append(sum_arr, np.array(
            [total, rewardSum, mean, counter, i, average, new_count]))
        total += rewardSum

        # checking parameter if not empty
        if not data:
            continue

        optimizer.zero_grad()

        scores = net(torch.FloatTensor(train))

        loss = crossEntropy(scores, torch.LongTensor(labels))

        loss.backward()

        optimizer.step()

        print(f"{i}: loss={loss.item()} Mean={mean} Sum={rewardSum} Cumulative={total}")

        counter += 1

        if sto or cliff:
            if i > episode:
                print('Accomplished!')
                break
        else:
            if mean > 0.79:
                print('Accomplished!')
                break

    return sum_arr.reshape(-1, 7)
