import os
import time
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

        print("[INFO] Testing Montezuma's Revenge Simplified environment")
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

        print("[INFO] Testing Taxi Simplified environment")
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

    env = gym.make(name_id)

    model = PPO.load(weight_path)
    obs = env.reset()
    done = False

    while True:
        if done:
            obs = env.reset()
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        time.sleep(1)
        print(obs, rewards)
        env.render()

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