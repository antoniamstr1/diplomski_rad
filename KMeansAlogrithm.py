import numpy as np
import random
from skimage import io
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


class KMeansCustom:
    def __init__(self, n_clusters, data=None):
        
        self.n_clusters = n_clusters
        self.data = np.array(data) if data is not None else np.array([])

        self.clusters = None
        self.centroids = None

        self.centroid_history = []
        self.cluster_history = []
        
    #-------------------------------------------------- E U C L I D I A N   D I S T A N C E --------------------------------------------------#
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    #--------------------------------------------------  C E N T R O I D   I N I T  +  +  +  --------------------------------------------------#
    def initialize_centroids(self):
        centroids = [self.data[np.random.randint(0, len(self.data))]]
        for _ in range(1, self.n_clusters):
            distances = np.array([min([self.euclidean_distance(d, c)**2 for c in centroids]) for d in self.data])
            probs = distances / distances.sum()
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids.append(self.data[j])
                    break
        return np.array(centroids)
    
    #--------------------------------------------------  C E N T R O I D   I N I T --------------------------------------------------#
    def initialize_centroids_basic_algorithm(self):
        indices = np.random.choice(len(self.data), self.n_clusters, replace=False)
        centroids = self.data[indices]
        return centroids

    #--------------------------------------------------  A S S I G N  C L U S T E R S --------------------------------------------------#
    def assign_clusters(self):
        clusters = []
        for d in self.data:
            closest_centroid = np.argmin([self.euclidean_distance(d, c) for c in self.centroids])
            clusters.append(closest_centroid)
        self.clusters = np.array(clusters)
        return self

   #--------------------------------------------------  U P D A T E   C L U S T E R S --------------------------------------------------#
    def update_centroids(self):
        new_centroids = []
        for label in range(self.n_clusters):
            group = self.data[self.clusters == label]
            if len(group) == 0:
                new_centroid = self.data[np.random.randint(0, len(self.data))]
            else:
                new_centroid = np.mean(group, axis=0)
            new_centroids.append(new_centroid)
        self.centroids = np.array(new_centroids)
        return self

    def pipeline(self):
        self.centroids = self.initialize_centroids()
        old_centroids = self.centroids.copy()

        self.centroid_history = [old_centroids.copy()]  
        self.cluster_history = []

        self.counter = 50000
        while True and self.counter > 0:
            self.counter -= 1
            self.assign_clusters()
            self.cluster_history.append(self.clusters.copy()) 

            self.update_centroids()
            self.centroid_history.append(self.centroids.copy()) 

            if np.allclose(old_centroids, self.centroids):
                break

            old_centroids = self.centroids.copy()
        return self.clusters, self.centroid_history


   #--------------------------------------------------  S S E  --------------------------------------------------#
    def sse(self):
        sse = 0.0
        for i in range(len(self.data)):
            center = self.centroids[self.clusters[i]]
            sse += np.sum((self.data[i] - center) ** 2)
        return sse