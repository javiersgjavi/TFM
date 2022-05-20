import os
import gym
import argparse
import numpy as np
from stable_baselines3 import PPO
from envs.TaxiSimplified import TaxiSimplified
from envs.MontezumaSimplified import MontezumaSimplified
from stable_baselines3.common.env_util import make_vec_env


def main(args):
    env_id = int(args.environment[0])
    save_path = './weights/'

    if env_id == 0:

        print("[INFO] Using Montezuma's Revenge Simplified environment")
        goals = np.load('./goals/montezuma.npy')
        name_id = 'MontezumaSimplified-v0'
        weight_path = f'{save_path}montezuma'

        gym.envs.register(
            id=name_id,
            entry_point='envs.MontezumaSimplified:MontezumaSimplified',
            max_episode_steps=20,
            kwargs={
                'goals': goals,
                'goal_reward': 100,
                'death_penalty': -20,
                'episode_limit': 300,
                'margin': 5}
        )

    elif env_id == 1:

        print("[INFO] Using Taxi Simplified environment")
        goals = np.load('./goals/taxi.npy')
        name_id = 'TaxiSimplified-v0'
        weight_path = f'{save_path}taxi'

        gym.envs.register(
            id=name_id,
            entry_point='envs.TaxiSimplified:TaxiSimplified',
            max_episode_steps=20,
            kwargs={
                'goals': goals,
                'goal_reward': 100,
                'death_penalty': -20,
                'episode_limit': 12}
        )

    envs = make_vec_env(name_id, n_envs=10)

    model = PPO('MlpPolicy', envs, verbose=1)
    model.learn(total_timesteps=5 * 10 ** 5)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.save(weight_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--environment",
        nargs=1,
        default=['0'],
        choices=['0', '1'],
        help="Environment to use"
    )
    args = parser.parse_args()
    main(args)
