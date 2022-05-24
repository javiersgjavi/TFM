import gym
import numpy as np
from gym import spaces
from models.Kmeans import Kmeans
from models.Models_PPO import PPOAgent


class MontezumaHierarchical(gym.Env):
    def __init__(self, goals, margin, steps_kmeans, limit_same_position, steps_limit, goal_reward, death_penalty, buffer_size):
        super(MontezumaHierarchical, self).__init__()
        self.env = gym.make('MontezumaRevenge-ram-v4')
        self.buffer_size = buffer_size
        self.controller = PPOAgent(
            env_name='montezuma',
            num_states=self.env.observation_space.shape[0],
            num_actions=self.env.action_space.n,
            lr=0.003,
            epochs=10,
            batch_size=64,
            shuffle=True,
            buffer_size=self.buffer_size,
            loss_clipping=0.2,
            entropy_loss=0.001,
            gamma=0.99,
            lambda_=0.95,
            normalize=True
        )
        self.controller.load()
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
        self.steps_without_reward = 0
        self.steps_limit = steps_limit
        self.detected_goals = 0
        self.goal_reward = goal_reward
        self.death_penalty = death_penalty

    def step(self, action):

        self.current_goal = self.goals[action]

        state, reward, done, info = self.controller_act()

        self.last_state = state
        self.episode += 1

        print(self.episode, self.steps)

        return state, reward, done, info

    def controller_act(self):

        rewards = []
        done = False
        stop_controller = False

        i_state = self.get_intrinsic_state(self.last_state)
        self.last_position = self.get_position(i_state)
        while self.get_distance_goal(i_state) > self.margin and not done and not stop_controller:

            if self.step % self.buffer_size == 0 and self.steps != 0:
                self.controller.replay()

            action, action_onehot, prediction = self.controller.act(i_state)
            next_state, reward, done, info = self.env.step(action)
            self.last_info = info

            i_next_state = self.get_intrinsic_state(next_state)
            i_reward, stop_controller = self.get_intrinsic_reward(next_state, reward, done)

            self.controller.store(i_state, action_onehot, i_reward, i_next_state, done, prediction)

            i_state = i_next_state
            rewards.append(reward)
            self.store_experience_kmeans(next_state)
            self.train_kmeans()

        return next_state, np.sum(rewards), done, self.last_info

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
        return i_observation.reshape((-1, i_observation.shape[0]))

    def store_experience_kmeans(self, state):
        position = self.get_position(state)
        life = self.get_life(state)

        if self.life == life:
            self.kmeans.store_experience(position)
        else:
            self.life = life

    def train_kmeans(self):
        self.steps += 1
        if self.steps - self.steps_last_kmeans >= self.step_kmeans:
            print(f'Last goals: {self.goals}')
            self.goals = self.kmeans.fit(self.goals)
            print(f'New goals: {self.goals}')
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

    def check_anomaly(self, reward, position):
        if reward > 0:
            index = np.random.randint(len(self.goals))
            self.goals[index] = position
            self.detected_goals += 1
            print(f'{self.detected_goals} detected goals, new: {position}')

    def get_intrinsic_reward(self, observation, reward, done):
        life = self.get_life(observation)
        position = self.get_position(observation)
        distance = np.linalg.norm(self.current_goal - position)
        reward = np.min(reward, -1)
        self.steps_without_reward += 1

        if distance <= self.margin:
            reward = self.goal_reward
            done = True

        elif self.steps_without_reward == self.steps_limit:
            reward = self.death_penalty // 2
            done = True

        elif life == self.life - 1:
            reward = self.death_penalty

        return reward, done
