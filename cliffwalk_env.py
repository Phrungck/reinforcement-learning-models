import gym
from gym import spaces


class CliffWalking(gym.Env):
    def __init__(self):
        self.col = 12
        self.row = 4
        self.s = (3, 0)  # starting position
        self.g = (3, 11)  # goal of the problem
        self.state = (3, 0)  # vector
        # represents the location at bottom of grid
        self.cliff = [x for x in range(37, 47)]

        #
        self.observation_space = spaces.Discrete(self.row*self.col)

        self.action_space = spaces.Discrete(4)

    def observation(self, state):
        return state[0]*self.col + state[1]  # first vector number in [x,y] + y
    # represents the position in 4x12 grid

    def step(self, action):
        if action == 0:  # left
            self.state = (self.state[0], max(0, self.state[1]-1))
        elif action == 1:  # down
            self.state = (min(self.row-1, self.state[0]+1), self.state[1])
        elif action == 2:  # right
            self.state = (self.state[0], min(self.col-1, self.state[1]+1))
        elif action == 3:  # up
            self.state = (max(0, self.state[0]-1), self.state[1])
        else:
            raise Exception('Invalid action.')

        # reward elsewhere
        reward = 0
        obs = self.state[0]*self.col + self.state[1]

        if obs in self.cliff:
            reward = -100  # reward if state is in cliff
            done = True
        elif obs == 47:
            reward = 0  # reward if state is in goal
            done = True
        else:
            reward = -1  # reward everywhere else
            done = False

        return self.observation(self.state), reward, done, _

    def reset(self):  # reset the environment
        self.state = self.s
        return self.observation(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
