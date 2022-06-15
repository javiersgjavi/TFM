import os
import gym
import pickle
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class Montezuma:
    def __init__(self, num_goals=10, margin=5, steps_kmeans=10 ** 5):

        self.num_goals = num_goals
        self.env_controller = 'MontezumaSimplified-v0'
        self.env_meta_controller = 'MontezumaHierarchical-v0'
        self.goals_detected = np.zeros((self.num_goals, 2))
        self.path_w_controller = './weights_controller/montezuma/'
        self.path_w_meta_controller = './weights_metacontroller/montezuma/'
        self.path_goals_detected = './goals_detected/'
        self.goals_detected = None
        self.margin = margin
        self.steps_kmeans = steps_kmeans
        self.kmeans = None

    def intrinsic_learning(self, steps, n_envs=2):
        print('[INFO] Starting intrinsic learning...')

        gym.envs.register(
            id=self.env_controller,
            entry_point='envs.MontezumaSimplified:MontezumaSimplified',
            max_episode_steps=500,
            kwargs={
                'goals': self.generate_random_goals(),
                'goal_reward': 200,
                'death_penalty': -20,
                'episode_limit': 500,
                'margin': self.margin,
                'steps_kmeans': self.steps_kmeans
            }
        )

        envs = make_vec_env(self.env_controller, n_envs=n_envs)
        model = PPO('MlpPolicy', envs, verbose=1)
        model.learn(total_timesteps=steps)

        if not os.path.exists(self.path_w_controller):
            os.makedirs(self.path_w_controller)
        model.save(f'{self.path_w_controller}best_model')

    def unified_learning(self, steps, load_kmeans=False):
        print('[INFO] Starting unified learning...')
        self.goals_detected = np.load(f'{self.path_goals_detected}trained_intrinsic_montezuma.npy')
        if load_kmeans:
            if self.kmeans is None:
                with open('./goals_detected/kmeans_memory', 'rb') as file:
                    self.kmeans = pickle.load(file)

        gym.envs.register(
            id=self.env_meta_controller,
            entry_point='envs.MontezumaHierarchical:MontezumaHierarchical',
            max_episode_steps=self.num_goals,
            kwargs={
                'goals': self.goals_detected,
                'margin': self.margin,
                'steps_kmeans': self.steps_kmeans,
                'limit_same_position': 20,
                'steps_limit': 500,
                'goal_reward': 100,
                'death_penalty': -20,
                'buffer_size': 2048,
                'kmeans': self.kmeans
            }
        )

        env = gym.make(self.env_meta_controller)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=steps)

        if not os.path.exists(self.path_w_meta_controller):
            os.makedirs(self.path_w_meta_controller)

        model.save(f'{self.path_w_meta_controller}best_model')

    def train(self, steps=7.8 * 10 ** 5, episodes=5 * 10 ** 5):
        self.intrinsic_learning(steps)
        self.unified_learning(episodes)

        if not os.path.exists('./goals_detected/'):
            os.makedirs('./goals_detected/')

        with open('./goals_detected/kmeans_memory', 'wb') as file:
            pickle.dump(self.kmeans, file)

        np.save('./goals_detected/trained_goals.npy', self.goals_detected)

    def generate_random_goals(self):
        goals = np.zeros((self.num_goals, 2))
        for i in range(self.num_goals):
            x = np.random.randint(255)
            y = np.random.randint(255)

            goals[i] = np.array([x, y])

        return goals
