import argparse
import gym
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main(args):
    job = int(args.job[0])
    episodes = 10 ** 4
    if job == 0 or job == 2:
        envs = make_vec_env('Taxi-v3', n_envs=10)
        model = PPO('MlpPolicy', envs, verbose=1)
        model.learn(total_timesteps=10 ** 6)
        if not os.path.exists('./weights_controller/basic_taxi/'):
            os.makedirs('./weights_controller/basic_taxi/')
        model.save('./weights_controller/basic_taxi/best_model')

    elif job == 1 or job == 2:
        model = PPO.load(f'./weights_controller/basic_taxi/best_model')
        env = gym.make('Taxi-v3')
        print(f'[INFO] Starting test with {episodes} episodes')

        episode, reward_ep, size_ep = 0, 0, 0

        ep_rew, ep_len = [], []
        state = env.reset()
        done = False

        while episode < episodes:
            if done:
                ep_len.append(size_ep)
                ep_rew.append(reward_ep)
                episode += 1
                reward_ep, size_ep = 0, 0
                state = env.reset()
                if episode % (episodes // 10) == 0:
                    print(f'Episode: {episode}')

            action, _state = model.predict(state)
            state, reward, done, _, = env.step(action)
            reward_ep += reward
            size_ep += 1

        ep_len_mean = np.mean(ep_len)
        ep_len_std = np.std(ep_len)

        ep_rew_mean = np.mean(ep_rew)
        ep_rew_std = np.std(ep_rew)

        msg = f'[RESULTS TAXI] Rewards episode: {round(ep_rew_mean, 2)} ±{round(ep_rew_std, 2)} | ' \
              f'Longitude episode {round(ep_len_mean, 2)} ±{round(ep_len_std, 2)}'

        if not os.path.exists('./results/'):
            os.makedirs('./results/')

        with open('./results/taxi_base_PPO', 'w') as file:
            file.write(msg)
        print(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--job",
        nargs=1,
        default=['0'],
        choices=['0', '1', '2'],
        help="Job to do. 0:train, 1:test, 2:both"
    )
    args = parser.parse_args()
    main(args)
