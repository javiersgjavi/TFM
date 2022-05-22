import gym
import numpy as np
from gym import spaces


class TaxiSimplified(gym.Env):
    def __init__(self, num_goals, goal_reward, death_penalty, episode_limit):
        super(TaxiSimplified, self).__init__()

        self.num_goals = num_goals
        self.goals = self.generate_random_goals()
        self.current_goal = None
        self.steps_without_reward = 0
        self.goal_reward = goal_reward
        self.env = gym.make('Taxi-v3')
        self.death_penalty = death_penalty
        self.episode_limit = episode_limit
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=4, shape=(5,), dtype=np.uint8)
        self.steps = 0
        self.steps_last_change_goals = 0
        self.steps_to_change_goals = 10 ** 4

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        i_observation = self.get_intrinsic_state(observation)
        i_reward, done = self.calculate_reward(i_observation, action, done)
        self.change_goals()

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

    def change_goals(self):
        self.steps += 1
        if self.steps - self.steps_last_change_goals >= self.steps_to_change_goals:
            self.goals = self.generate_random_goals()
            self.steps_last_change_goals = self.steps

    def generate_random_goals(self):
        goals = np.zeros((self.num_goals * 2, 3))
        for i in range(0, 2 * self.num_goals, 2):
            x = np.random.randint(6)
            y = np.random.randint(6)

            goals[i] = np.array([x, y, 0])
            goals[i + 1] = np.array([x, y, 1])

        return goals
