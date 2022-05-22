import gym
import pickle
import numpy as np
from gym import spaces
from stable_baselines3 import PPO


class TaxiHierarchical(gym.Env):
    def __init__(self, goals):
        super(TaxiHierarchical, self).__init__()
        self.env = gym.make('Taxi-v3')
        self.controller = PPO.load('./weights_controller/taxi')
        self.goals = goals
        self.action_space = spaces.Discrete(len(self.goals))
        self.observation_space = spaces.Box(low=0, high=4, shape=(4,), dtype=np.uint8)
        self.last_state = None

    def step(self, action):
        goal = self.goals[action]
        state = self.last_state
        i_state = self.get_intrinsic_state(state, goal)
        done = False
        total_rewards = []
        action = -1
        while not (np.array_equal(state[:2], goal[:2]) and (action == 4 or action == 5)):
            action, _states = self.controller.predict(i_state)
            state, reward, done, info = self.env.step(action)
            state = self.decode(state)
            total_rewards.append(reward)
            i_state = self.get_intrinsic_state(state, goal)

        self.last_state = state

        return state, np.sum(total_rewards), done, info

    def reset(self):
        state = self.env.reset()
        state_decoded = self.decode(state)
        self.last_state = state_decoded
        return state_decoded

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def decode(self, state):
        return np.array(list(self.env.decode(state)))

    def get_intrinsic_state(self, state, goal):
        return np.concatenate([state[:2], goal], axis=0)
