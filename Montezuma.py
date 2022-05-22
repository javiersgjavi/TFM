import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class Montezuma:
    def __init__(self, num_goals=10):

        self.num_goals = num_goals
        self.env_controller = 'MontezumaSimplified-v0'
        self.env_meta_controller = 'MontezumaHierarchical-v0'
        self.goals_detected = np.zeros((self.num_goals, 2))
        self.path_w_controller = './weights_controller/'
        self.path_w_meta_controller = './weights_metacontroller/'

    def intrinsic_learning(self, steps, n_envs=10):
        print('[INFO] Starting intrinsic learning...')

        gym.envs.register(
            id=self.env_controller,
            entry_point='envs.MontezumaSimplified:MontezumaSimplified',
            max_episode_steps=20,
            kwargs={
                'goals': self.generate_random_goals(),
                'goal_reward': 100,
                'death_penalty': -20,
                'episode_limit': 12}
        )

        env = gym.make(self.env_controller)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=steps)

        if not os.path.exists(self.path_w_controller):
            os.makedirs(self.path_w_controller)

        model.save(f'{self.path_w_controller}/montezuma')

    def unified_learning(self, steps, n_envs=10):
        print('[INFO] Starting unified learning...')

        gym.envs.register(
            id=self.env_meta_controller,
            entry_point='envs.MontezumaHierarchical:MontezumaHierarchical',
            max_episode_steps=20,
            kwargs={'goals': self.goals_detected}
        )

        envs = gym.make(self.env_meta_controller, n_envs=n_envs)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=steps)

        if not os.path.exists(self.path_w_meta_controller):
            os.makedirs(self.path_w_meta_controller)

        model.save(f'{self.path_w_meta_controller}/montezuma')

    def train(self, steps=5 * 10 ** 5, episodes=5 * 10 ** 5, n_envs=10):
        self.intrinsic_learning(steps, n_envs)
        self.unified_learning(episodes)

        if not os.path.exists('./goals_detected/'):
            os.makedirs('./goals_detected/')

        np.save('./goals_detected/trained_goals.npy', self.goals_detected)

    def generate_random_goals(self):
        goals = np.zeros((self.num_goals, 2))
        for i in range(self.num_goals):
            x = np.random.randint(255)
            y = np.random.randint(255)

            goals[i] = np.array([x, y])

        return goals
