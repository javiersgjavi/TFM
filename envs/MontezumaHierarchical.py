import gym
import numpy as np
from gym import spaces
from models.Kmeans import Kmeans


class MontezumaHierarchical(gym.Env):
    def __init__(self, goals, margin, steps_kmeans):
        super(MontezumaIntrinsic, self).__init__()
        self.env = gym.make('MontezumaRevenge-ram-v4')
        self.controller = PPO.load('./weights_controller/montezuma')
        self.goals = goals
        self.current_goal = None
        self.action_space = Discrete(len(self.goals))
        self.observation_space = self.env.observation_space.n
        self.margin = margin
        self.last_state = None
        self.step_kmeans = memory_kmeans
        self.kmeans = Kmeans(k=len(self.goals), memory_size=10**6)
        self.life = None
        self.steps_last_kmeans = 0
        self.steps = 0

    def step(self, action):

        goal = self.goals[action]
        state = self.last_state
        rewards = []
        i_state = self.get_intrinsic_state(state, goal)

        while self.get_distance_goal(i_state) > self.margin or done:
            action = self.controller.predict(i_state)
            state, reward, done, info = self.env.step(action)
            rewards.append(reward)

            i_state = self.get_intrinsic_state(observation, goal)
            life = self.get_life(state)
            if self.life == state:
                self.kmeans.store(self.get_position(state))

            self.life = life

            self.train_kmeans()

        self.last_state = state
        return state, np.sum(rewards), done, info

    def reset(self):
        observation = self.env.reset()
        self.last_state = observation
        self.life = self.get_life(observation)
        return observation

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

    def get_distance_goal(self, i_state):
        character_pos = self.get_position(i_state)
        goal_pos = i_state[-2:]
        return np.linalg.norm(goal_pos - character_pos)

    def get_intrinsic_state(self, observation):
        i_observation = np.concatenate([observation, self.current_goal], axis=0)
        return i_observation

    def train_kmeans(self):
        self.steps_last_kmeans += 1
        if self.steps - self.steps_last_kmeans >= self.step_kmeans:
            self.goals = self.kmeans.fit(self.goals)
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
