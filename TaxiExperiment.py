import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class Taxi:
    def __init__(self, num_goals=4):

        self.num_goals = num_goals
        self.env_controller = 'TaxiSimplified-v0'
        self.env_meta_controller = 'TaxiHierarchical-v0'
        self.goals_detected = []
        self.path_w_controller = './weights_controller/'
        self.path_w_meta_controller = './weights_metacontroller/'

    def intrinsic_learning(self, steps, n_envs=10):
        print('[INFO] Starting intrinsic learning...')

        gym.envs.register(
            id=self.env_controller,
            entry_point='envs.TaxiSimplified:TaxiSimplified',
            max_episode_steps=20,
            kwargs={
                'num_goals': 4,
                'goal_reward': 100,
                'death_penalty': -20,
                'episode_limit': 12}
        )

        envs = make_vec_env(self.env_controller, n_envs=n_envs)
        model = PPO('MlpPolicy', envs, verbose=1)
        model.learn(total_timesteps=steps)

        if not os.path.exists(self.path_w_controller):
            os.makedirs(self.path_w_controller)

        model.save(f'{self.path_w_controller}/taxi')

    def unified_learning(self, steps, n_envs=10):
        self.explore_goals()

        print('[INFO] Starting unified learning...')

        gym.envs.register(
            id=self.env_meta_controller,
            entry_point='envs.TaxiHierarchical:TaxiHierarchical',
            max_episode_steps=2,
            kwargs={'goals': self.goals_detected}
        )

        envs = make_vec_env(self.env_meta_controller, n_envs=n_envs)
        model = PPO('MlpPolicy', envs, verbose=1)
        model.learn(total_timesteps=steps)

        if not os.path.exists(self.path_w_meta_controller):
            os.makedirs(self.path_w_meta_controller)

        model.save(f'{self.path_w_meta_controller}/taxi')

    def explore_goals(self):
        print('[INFO] Starting the exploration of the environment to find the goals')
        goals = []
        env = gym.make('Taxi-v3')

        _ = env.reset()
        done = False

        while len(goals) < self.num_goals:
            if done:
                _ = env.reset()
            action = np.random.randint(6)
            obs, rewards, done, info = env.step(action)
            position = list(env.decode(obs))[:2]

            if rewards > 0 and self.check_anomaly(position, goals):
                goals.append(position)

        for i in range(len(goals)):
            x, y = goals[i][0], goals[i][1]

            self.goals_detected.append(np.array([x, y, 0]))
            self.goals_detected.append(np.array([x, y, 1]))

        self.goals_detected = np.array(self.goals_detected)
        print(f'[INFO] Detected goals: {goals}')
        print(self.goals_detected, self.goals_detected.shape)

    def train(self, steps=5 * 10 ** 5, episodes=5 * 10 ** 5, n_envs=10):
        self.intrinsic_learning(steps, n_envs)
        self.unified_learning(episodes)

        if not os.path.exists('./goals_detected/'):
            os.makedirs('./goals_detected/')

        np.save('./goals_detected/trained_goals.npy', self.goals_detected)

    def check_anomaly(self, position, goals):
        res = True
        for i in range(len(goals)):
            goal = goals[i]
            if position[0] == goal[0] and position[1] == goal[1]:
                res = False
                break
        return res
