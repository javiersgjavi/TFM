import gym
import numpy as np
from gym import spaces
from models.Kmeans import Kmeans
from stable_baselines3 import PPO


class MontezumaHierarchical(gym.Env):
    def __init__(self, goals, margin, steps_kmeans, limit_same_position):
        super(MontezumaHierarchical, self).__init__()
        self.env = gym.make('MontezumaRevenge-ram-v4')
        self.controller = PPO.load('./weights_controller/montezuma')
        self.goals = goals
        self.current_goal = None
        self.action_space = spaces.Discrete(len(self.goals))
        self.observation_space = self.env.observation_space
        self.margin = margin
        self.last_state = None
        self.last_info = None
        self.step_kmeans = steps_kmeans
        self.kmeans = Kmeans(k=len(self.goals), memory_size=10 ** 6)
        self.life = None
        self.steps_last_kmeans = 0
        self.steps = 0
        self.episode = 0
        self.last_position = None
        self.limit_same_position = limit_same_position
        self.steps_same_position = 0

    def step(self, action):

        self.current_goal = self.goals[action]
        state = self.last_state
        rewards = []
        i_state = self.get_intrinsic_state(state)
        done = False
        self.last_position = self.get_position(state)
        stop_action = False
        while self.get_distance_goal(i_state) > self.margin and not done and not stop_action:
            action, _states = self.controller.predict(i_state)
            state, reward, done, info = self.env.step(action)
            position = self.get_position(state)
            reward, stop_action = self.check_same_position(position, reward)
            rewards.append(reward)

            i_state = self.get_intrinsic_state(state)
            life = self.get_life(state)
            self.last_info = info

            # print(self.get_position(position), self.current_goal, life, done, info)

            if self.life == life:
                self.kmeans.store_experience(position)
            else:
                self.life = life

            self.train_kmeans()

        self.last_state = state
        self.episode += 1
        print(self.episode)

        return state, np.sum(rewards), done, self.last_info

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

    def check_same_position(self, position, reward):
        stop_action = False

        if position[0] == self.last_position[0] and position[1] == self.last_position[1]:
            self.steps_same_position += 1
        else:
            self.steps_same_position = 0
            self.last_position = position

        if self.steps_same_position >= self.limit_same_position:
            reward = -10
            stop_action = True
            self.steps_same_position = 0

        return reward, stop_action
