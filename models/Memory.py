import numpy as np


class MemoryKmeans:
    def __init__(self, size):
        self.size = size
        self.states = np.zeros((self.size, 2))
        self.index = 0

    def store(self, state):
        i = self.index % self.size
        self.states[i, :] = state
        self.index += 1

    def get_stored_experiences(self):
        return self.states


class MemoryPPO:
    def __init__(self, memory_size, input_shape, action_size):
        self.states = np.zeros((memory_size, input_shape))
        self.next_states = np.zeros((memory_size, input_shape))
        self.actions = np.zeros((memory_size, action_size))
        self.rewards = np.zeros(memory_size)
        self.predictions = np.zeros((memory_size, action_size))
        self.done = np.zeros(memory_size, dtype=bool)
        self.size = memory_size
        self.index = 0

    def store(self, state, action_onehot, reward, state_next, done, prediction):
        self.states[self.index] = state
        self.actions[self.index] = action_onehot
        self.rewards[self.index] = reward
        self.next_states[self.index] = state_next
        self.done[self.index] = done
        self.predictions[self.index] = prediction
        self.index += 1

    def sample_memory(self):
        index = np.random.shuffle(np.arange(self.size))
        return self.states[index][0], self.actions[index][0], self.rewards[index][0], self.next_states[index][0], self.predictions[index][0], self.done[index][0]









