import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


class Taxi:
    def __init__(self, num_goals=4):

        self.num_goals = num_goals
        self.env_controller = 'TaxiSimplified-v0'
        self.env_meta_controller = 'TaxiHierarchical-v0'
        self.goals_detected = []
        self.path_w_controller = './weights_controller/taxi/'
        self.path_w_meta_controller = './weights_metacontroller/taxi/'

        self.path_goals_detected = './goals_detected/'

    def intrinsic_learning(self, steps, n_envs=10):
        print('[INFO] Starting intrinsic learning...')
        if not os.path.exists(self.path_w_controller):
            os.makedirs(self.path_w_controller)

        gym.envs.register(
            id=self.env_controller,
            entry_point='envs.TaxiSimplified:TaxiSimplified',
            max_episode_steps=20,
            kwargs={
                'goal_reward': 100,
                'death_penalty': -20,
                'episode_limit': 12}
        )

        envs = make_vec_env(self.env_controller, n_envs=n_envs)
        model = PPO('MlpPolicy', envs, n_steps=1024, verbose=1)
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=70, verbose=1)
        eval_callback = EvalCallback(gym.make(self.env_controller), best_model_save_path=self.path_w_controller,
                                     log_path=self.path_w_controller, callback_on_new_best=callback_on_best, verbose=1)

        model.learn(total_timesteps=steps, callback=eval_callback)

    def unified_learning(self, steps, n_envs=10):
        self.explore_goals()

        print('[INFO] Starting unified learning...')
        if not os.path.exists(self.path_w_meta_controller):
            os.makedirs(self.path_w_meta_controller)

        gym.envs.register(
            id=self.env_meta_controller,
            entry_point='envs.TaxiHierarchical:TaxiHierarchical',
            max_episode_steps=2,
            kwargs={'goals': self.goals_detected}
        )

        envs = make_vec_env(self.env_meta_controller, n_envs=n_envs)
        model = PPO('MlpPolicy', envs, n_steps=512, verbose=1)
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=7, verbose=1)
        eval_callback = EvalCallback(envs, best_model_save_path=self.path_w_meta_controller,
                                     log_path=self.path_w_meta_controller, callback_on_new_best=callback_on_best, verbose=0)
        model.learn(total_timesteps=steps, callback=eval_callback)

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

        if not os.path.exists(self.path_goals_detected):
            os.makedirs(self.path_goals_detected)

        np.save(f'{self.path_goals_detected}trained_taxi.npy', self.goals_detected)

    def train(self, steps=7 * 10 ** 5, episodes=int(1.6 * 10 ** 5), n_envs=10):
        self.intrinsic_learning(steps, n_envs)
        self.unified_learning(episodes)

    def watch(self):

        model, env, done = self.start_env_to_play()

        state = env.reset()

        while True:
            if done:
                state = env.reset()

            action, _states = model.predict(state)
            state, reward, done, info = env.render_step(action)
            print(action)

    def test(self, episodes=10 ** 3):
        print(f'[INFO] Starting test with {episodes} episodes')
        model, env, done = self.start_env_to_play()

        episode, reward_ep, size_ep = 0, 0, 0

        ep_rew, ep_len = [], []
        state = env.reset()

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
            state, reward, done, _, steps = env.test_step(action)
            reward_ep += reward
            size_ep += steps

        ep_len_mean = np.mean(ep_len)
        ep_len_std = np.std(ep_len)

        ep_rew_mean = np.mean(ep_rew)
        ep_rew_std = np.std(ep_rew)

        msg = f'[RESULTS TAXI] Rewards episode: {round(ep_rew_mean, 2)} ??{round(ep_rew_std, 2)} | ' \
              f'Longitude episode {round(ep_len_mean, 2)} ??{round(ep_len_std, 2)}'

        if not os.path.exists('./results/'):
            os.makedirs('./results/')

        with open('./results/taxi', 'w') as file:
            file.write(msg)
        print(msg)

    def start_env_to_play(self):
        goals = np.load('./goals_detected/trained_taxi.npy')

        gym.envs.register(
            id=self.env_meta_controller,
            entry_point='envs.TaxiHierarchical:TaxiHierarchical',
            max_episode_steps=2,
            kwargs={'goals': goals}
        )

        model = PPO.load(f'{self.path_w_meta_controller}best_model')
        env = gym.make(self.env_meta_controller)
        done = False
        return model, env, done

    def check_anomaly(self, position, goals):
        res = True
        for i in range(len(goals)):
            goal = goals[i]
            if position[0] == goal[0] and position[1] == goal[1]:
                res = False
                break
        return res
