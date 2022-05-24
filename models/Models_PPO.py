import os

import random
import gym
import pylab
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# tf.config.experimental_run_functions_eagerly(True)
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import copy
from models.Memory import MemoryPPO


# Create the actor used to select the action given an state
class Actor_Model:
    def __init__(self, num_states, num_actions, lr, optimizer, entropy_loss, loss_clipping):
        self.num_actions = num_actions
        self.entropy_loss = entropy_loss
        self.loss_clipping = loss_clipping
        X_input = Input(num_states)

        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)

        # Softmax as there are different probabilities depending on the action
        output = Dense(self.num_actions, activation="softmax")(X)

        # Compile the model with the custom loss
        self.model = Model(inputs=X_input, outputs=output)
        self.model.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))

    # Custom loss functions for the PPO
    def ppo_loss(self, y_true, y_pred):
        # Unpack the elements given in the true label
        advantages, true_label, actions = y_true[:, :1], y_true[:, 1:1 + self.num_actions], y_true[:,
                                                                                            1 + self.num_actions:]

        prob = actions * y_pred
        old_prob = actions * true_label

        ratio = K.exp(K.log(prob + 1e-10) - K.log(old_prob + 1e-10))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = self.entropy_loss * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss


# Create the critic which will criticise how the actor is performing
class Critic_Model:
    def __init__(self, num_states, lr, optimizer):
        X_input = Input(num_states)

        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X_input)
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        # Linear output to know how good the action is
        value = Dense(1)(X)

        # Compile it with mse loss and gradient descent
        self.model = Model(inputs=X_input, outputs=value)
        self.model.compile(loss='mse', optimizer=optimizer(lr=lr))


# Combine both Actor and Critic to create the agent
class PPOAgent:
    def __init__(self, env_name, num_states, num_actions, lr, epochs, batch_size, shuffle, buffer_size, loss_clipping,
                 entropy_loss, gamma, lambda_, normalize):
        # Environment parameters
        self.env_name = env_name
        self.episode = 0  # used to track current number episoded since start
        self.max_average = 0  # record max average reached
        self.num_actions = num_actions
        self.num_states = num_states
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lambda_ = lambda_
        self.normalize = normalize
        self.memory = MemoryPPO(memory_size=self.buffer_size, input_shape=self.num_states, action_size=self.num_actions)

        # Used to plot a grapgh of the train process
        self.scores_, self.average_ = [], []

        # Create Actor-Critic network models
        self.Actor = Actor_Model(num_states=self.num_states, num_actions=self.num_actions, lr=lr, optimizer=Adam,
                                 entropy_loss=entropy_loss, loss_clipping=loss_clipping)
        self.Critic = Critic_Model(num_states=self.num_states, lr=lr, optimizer=Adam)

        # Names for the models
        self.Actor_name = f"./weights_controller/{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"./weights_controller/{self.env_name}_PPO_Critic.h5"

    # Get the action given the current state
    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.model.predict(state)[0]

        # Probability based to choose the action
        action = np.random.choice(self.num_actions, p=prediction)
        action_onehot = np.zeros([self.num_actions])
        action_onehot[action] = 1
        return action, action_onehot, prediction

    # Generalized Advantage Estimation implemented in the original paper
    def get_gaes(self, rewards, dones, values, next_values):
        # Dones are used to track when is the final step of an episode, so next values are no applied
        deltas = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * self.gamma * self.lambda_ * gaes[t + 1]

        target = gaes + values
        if self.normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self):
        # Reshape memory to appropriate shape for training

        states, actions, rewards, next_states, predictions, dones = self.memory.sample_memory()

        # Get Critic network predictions for state and next state
        values = self.Critic.model.predict(states)
        next_values = self.Critic.model.predict(next_states)

        # Get the advantage
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # Stack info to unpack it in the custom loss
        y_true = np.hstack([advantages, predictions, actions])

        # Training Actor and Critic networks
        a_loss = self.Actor.model.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle,
                                      batch_size=self.batch_size)
        c_loss = self.Critic.model.fit(states, target, epochs=self.epochs, verbose=0, shuffle=self.shuffle,
                                       batch_size=self.batch_size)
        self.memory = MemoryPPO(memory_size=self.buffer_size, input_shape=self.num_states, action_size=self.num_actions)

    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.model.save_weights(self.Actor_name)
        self.Critic.model.save_weights(self.Critic_name)

    def store(self, states, actions, rewards, next_states, dones, predictions):
        self.memory.store(states, actions, rewards, next_states, dones, predictions)

    def run_batch(self):  # train every self.Training_batch episodes
        global LR
        state = self.env.reset()
        state = np.reshape(state, [1, self.num_states])
        done, score = False, 0
        finished = False
        while finished == False:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            for t in range(self.buffer_size):
                # self.env.render()

                # Actor picks an action
                action, action_onehot, prediction = self.act(state)

                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)

                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.num_states]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = np.reshape(next_state, [1, self.num_states])
                score += reward
                if done:
                    self.episode += 1
                    self.scores_.append(score)
                    # average, SAVING = self.PlotModel(score, self.episode)
                    # print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, EPISODES, score, average, SAVING))
                    if self.episode >= 100:
                        average = sum(self.scores_[-100:]) / 100
                        print('Episode: {:>5}\t\tscore: {:>7.2f}\t\taverage: {:>7.2f}'.format(self.episode, score,
                                                                                              average))
                        if average > self.max_average:
                            self.max_average = average
                            if self.max_average > 150:
                                self.save()
                            LR *= 0.95
                            K.set_value(self.Actor.model.optimizer.learning_rate, LR)
                            K.set_value(self.Critic.model.optimizer.learning_rate, LR)

                        if average > 200:
                            plt.plot(self.scores_)
                            plt.xlabel("Episode")
                            plt.ylabel("Score")
                            finished = True
                            break

                    else:
                        print('Episode: {:>5}\t\tscore: {:>7.2f}\t\taverage: {:>7.2f}'.format(self.episode, score,
                                                                                              sum(self.scores_) / len(
                                                                                                  self.scores_)))

                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape(state, [1, NUM_STATES])

            self.replay(states, actions, rewards, predictions, dones, next_states)
            if self.episode >= EPISODES:
                break
        self.env.close()
