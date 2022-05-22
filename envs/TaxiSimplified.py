import gym
import numpy as np
from gym import spaces


class TaxiSimplified(gym.Env):
    def __init__(self, goals, goal_reward, death_penalty, episode_limit):
        super(TaxiSimplified, self).__init__()

        self.goals = goals
        self.current_goal = None
        self.steps_without_reward = 0
        self.goal_reward = goal_reward
        self.env = gym.make('Taxi-v3')
        self.death_penalty = death_penalty
        self.episode_limit = episode_limit
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=4, shape=(5,), dtype=np.uint8)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        i_observation = self.get_intrinsic_state(observation)
        i_reward, done = self.calculate_reward(i_observation, action, done)

        return i_observation, i_reward, done, info

    def reset(self):
        random_goal = np.random.randint(0, len(self.goals))
        self.current_goal = self.goals[random_goal]
        observation = self.env.reset()
        i_observation = self.get_intrinsic_state(observation)
        return i_observation

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def get_intrinsic_state(self, observation):
        observation = np.array(list(self.env.decode(observation)))[:2]
        i_observation = np.concatenate([observation, self.current_goal], axis=0)
        return i_observation

    def calculate_reward(self, observation, action, done):
        in_place = np.array_equal(observation[:2], self.current_goal[:2])
        pick = self.current_goal[-1] == 0
        drop = self.current_goal[-1] == 1
        reward = -1
        self.steps_without_reward += 1

        if in_place and ((pick and action == 4) or (drop and action == 5)):
            done = True
            reward = self.goal_reward

        elif action == 4 or action == 5:
            done = True
            reward = self.death_penalty

        elif self.steps_without_reward == self.episode_limit:
            done = True
            reward = self.death_penalty // 2

        if done:
            self.steps_without_reward = 0

        return reward, done



