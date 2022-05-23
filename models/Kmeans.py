import numpy as np
from sklearn.cluster import KMeans
from models.Memory import MemoryKmeans


class Kmeans:
    def __init__(self, k=10, max_iterations=100, memory_size=10 ** 6):
        self.k = k
        self.max_iterations = max_iterations
        self.memory = MemoryKmeans(memory_size)

    def store_experience(self, position):
        self.memory.store(position)

    def normalize_experiences(self, experience_set, goals):
        maxs = 255
        mins = 0

        experience_set = np.array(experience_set)

        denominator = maxs - mins
        experience_set = (experience_set - mins) / denominator

        goals = (goals - mins) / denominator

        return experience_set, goals

    def fit(self, goals):
        experience_set = self.memory.get_stored_experiences()

        experience_set, centroids = self.normalize_experiences(experience_set, goals)

        kmeans = KMeans(n_clusters=self.k, max_iter=self.max_iterations, init=centroids, n_init=1)

        kmeans.fit(experience_set)

        clusters_centers = kmeans.cluster_centers_

        denorm_clusters_centers = clusters_centers * (255 - 0) + 0
        denorm_clusters_centers = denorm_clusters_centers.astype(int)

        return denorm_clusters_centers
