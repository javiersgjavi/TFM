import numpy as np


class MemoryKmeans:
    def __init__(self, size):
        self.size = size
        self.states = np.zeros((self.size, 2))
        self.index = 0

    def store(self, state):
        i = self.index % self.size
        self.states[self.index, :] = state
        self.index += 1

    def get_stored_experiences(self):
        return self.states
