import gym
import numpy as np
from gym import spaces
from models.Kmeans import Kmeans


class MontezumaSimplified(gym.Env):
    def __init__(self, goals, goal_reward, death_penalty, episode_limit, margin, steps_kmeans):
        super(MontezumaSimplified, self).__init__()
        self.env = gym.make('MontezumaRevenge-ram-v4')
        self.goals = goals
        self.current_goal = None
        self.goal_reward = goal_reward
        self.death_penalty = death_penalty
        self.episode_limit = episode_limit
        self.steps_without_reward = 0
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=(130,), dtype=np.uint8)
        self.life = 0
        self.margin = margin
        self.steps_last_kmeans = 0
        self.kmeans = Kmeans(k=len(self.goals), memory_size=10 ** 6)
        self.steps_kmeans = steps_kmeans
        self.steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        i_observation = self.get_intrinsic_state(observation)
        i_reward, done = self.get_intrinsic_reward(i_observation, reward, done)
        if i_reward >= 0:
            self.kmeans.store_experience(self.get_position(observation))
        self.train_kmeans()
        return i_observation, i_reward, done, info

    def reset(self):
        random_goal = np.random.randint(0, len(self.goals))
        self.current_goal = self.goals[random_goal]
        observation = self.env.reset()
        self.life = self.get_life(observation)
        i_observation = self.get_intrinsic_state(observation)
        return i_observation

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def get_life(self, observation):
        return observation[58]

    def get_position(self, observation):
        x = observation[42]
        y = observation[43]
        return np.array([x, y])

    def get_intrinsic_state(self, observation):
        i_observation = np.concatenate([observation, self.current_goal], axis=0)
        return i_observation

    def train_kmeans(self):
        self.steps += 1
        if self.steps - self.steps_last_kmeans >= self.steps_kmeans:
            print(f'Last goals: {self.goals}')
            self.goals = self.kmeans.fit(self.goals, self.steps_last_kmeans == 0)
            print(f'New goals: {self.goals}')
            self.steps_last_kmeans = self.steps

    def get_intrinsic_reward(self, observation, reward, done):
        life = self.get_life(observation)
        position = self.get_position(observation)
        distance = np.linalg.norm(self.current_goal - position)
        reward = np.min(reward, -1)
        self.steps_without_reward += 1

        if distance <= self.margin:
            reward = self.goal_reward
            done = True

        elif self.steps_without_reward == self.episode_limit:
            reward = self.death_penalty // 2
            done = True

        elif life == self.life - 1:
            reward = self.death_penalty

        return reward, done

    def get_goals(self):
        return self.goals
