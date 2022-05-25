import time
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
        self.current_goal = None
        self.action_space = spaces.Discrete(len(self.goals))
        self.observation_space = spaces.Box(low=0, high=4, shape=(4,), dtype=np.uint8)
        self.last_state = None

    def step(self, action):
        i_state, done, total_rewards, action = self.prepare_step(action)

        while self.check_stop_condition(action):
            i_state, total_rewards, done, info, action = self.controller_act(i_state, total_rewards)

        return self.last_state, np.sum(total_rewards), done, info

    def render_step(self, action):
        i_state, done, total_rewards, action = self.prepare_step(action)

        while self.check_stop_condition(action):
            i_state, total_rewards, done, info, action = self.controller_act(i_state, total_rewards)
            self.render()
            print(i_state[:2], self.current_goal,  total_rewards[-1])
            time.sleep(0.1)

        return self.last_state, np.sum(total_rewards), done, info

    def test_step(self, action):
        i_state, done, total_rewards, action = self.prepare_step(action)
        steps = 0
        while self.check_stop_condition(action):
            i_state, total_rewards, done, info, action = self.controller_act(i_state, total_rewards)
            steps += 1

        return self.last_state, np.sum(total_rewards), done, info, steps

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

    def get_intrinsic_state(self, state):
        return np.concatenate([state[:2], self.current_goal], axis=0)

    def prepare_step(self, action):
        self.current_goal = self.goals[action]
        i_state = self.get_intrinsic_state(self.last_state)
        done = False
        total_rewards = []
        action = -1
        return i_state, done, total_rewards, action

    def controller_act(self, i_state, total_rewards):
        action, _states = self.controller.predict(i_state)
        state, reward, done, info = self.env.step(action)
        state = self.decode(state)
        total_rewards.append(reward)
        i_state = self.get_intrinsic_state(state)
        self.last_state = state
        return i_state, total_rewards, done, info, action

    def check_stop_condition(self, action):
        return not (np.array_equal(self.last_state[:2], self.current_goal[:2]) and (action == 4 or action == 5))

