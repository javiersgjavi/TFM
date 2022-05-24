import os
import gym
import numpy as np

from stable_baselines3 import PPO
from models.Models_PPO import PPOAgent
from stable_baselines3.common.env_util import make_vec_env


class Montezuma:
    def __init__(self, num_goals=10, margin=5, steps_kmeans=10 ** 5):

        self.num_goals = num_goals
        self.env_controller = 'MontezumaSimplified-v0'
        self.env_meta_controller = 'MontezumaHierarchical-v0'
        self.goals_detected = np.zeros((self.num_goals, 2))
        self.path_w_controller = './weights_controller/'
        self.path_w_meta_controller = './weights_metacontroller/'
        self.path_goals_detected = './goals_detected/'
        self.goals_detected = None
        self.margin = margin
        self.steps_kmeans = steps_kmeans

    def intrinsic_learning(self, steps):
        print('[INFO] Starting intrinsic learning...')

        gym.envs.register(
            id=self.env_controller,
            entry_point='envs.MontezumaSimplified:MontezumaSimplified',
            max_episode_steps=500,
            kwargs={
                'goals': self.generate_random_goals(),
                'goal_reward': 100,
                'death_penalty': -20,
                'episode_limit': 500,
                'margin': self.margin,
                'steps_kmeans': self.steps_kmeans
            }
        )

        buffer_size = 300
        env = gym.make(self.env_controller)
        model = PPOAgent(
            env_name='montezuma',
            num_states=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            lr=0.003,
            epochs=10,
            batch_size=64,
            shuffle=True,
            buffer_size=buffer_size,
            loss_clipping=0.2,
            entropy_loss=0.001,
            gamma=0.99,
            lambda_=0.95,
            normalize=True
        )

        done = True
        rewards, long = [], []
        episode_len, reward_ep = 0, 0

        for step in range(steps):

            if step % buffer_size == 0 and step != 0:
                model.replay()

            if done:
                state = env.reset()
                rewards.append(reward_ep)
                long.append(episode_len)
                episode_len, reward_ep = 0, 0
                print(f'step: {step} | ep_rew_mean: {round(np.mean(rewards[-100:]))} | ep_len_mean: {round(np.mean(long[-100:]))}')

            state = state.reshape((-1, state.shape[0]))
            action, action_onehot, prediction = model.act(state)
            next_state, reward, done, info = env.step(action)
            model.store(state, action_onehot, reward, next_state, done, prediction)
            state = next_state
            episode_len += 1
            reward_ep += reward

        if not os.path.exists(self.path_w_controller):
            os.makedirs(self.path_w_controller)

        model.save()

        if not os.path.exists(self.path_goals_detected):
            os.makedirs(self.path_goals_detected)

        np.save(f'{self.path_goals_detected}trained_intrinsic_montezuma.npy', env.get_goals())

    def unified_learning(self, steps):
        print('[INFO] Starting unified learning...')
        self.goals_detected = np.load(f'{self.path_goals_detected}trained_intrinsic_montezuma.npy')

        gym.envs.register(
            id=self.env_meta_controller,
            entry_point='envs.MontezumaHierarchical:MontezumaHierarchical',
            max_episode_steps=self.num_goals,
            kwargs={
                'goals': self.goals_detected,
                'margin': self.margin,
                'steps_kmeans': self.steps_kmeans,
                'limit_same_position': 20,
            }
        )

        env = gym.make(self.env_meta_controller)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=steps)

        if not os.path.exists(self.path_w_meta_controller):
            os.makedirs(self.path_w_meta_controller)

        model.save(f'{self.path_w_meta_controller}/montezuma')

    def train(self, steps=5 * 10 ** 5, episodes=5 * 10 ** 5):
        self.intrinsic_learning(steps)
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
