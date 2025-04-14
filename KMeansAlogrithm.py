import numpy as np
import random

class KMeansCustom:
    def __init__ (self, n_clusters, data):
        self.n_clusters = n_clusters #broj grupa, centroida
        self.data = data
        
    def euclidean_distance(self, a,b):
        return np.sqrt(np.sum((a - b) ** 2))


    """ sets self.clusters """
    def assign_clusters(self):
        clusters = []
        for d in self.data:
            # dodajemo index središta do kojeg teba najmanje
            closest_centroid = np.argmin([self.euclidean_distance(d, c) for c in self.centroids])
            clusters.append(closest_centroid)
        self.clusters = clusters
        return self
    
    """ updates centroids """
    def update_centroids(self):
        unique_labels =list(range(self.n_clusters))
        means = []
        
        for label in unique_labels:
            group = self.data[np.array(self.clusters) == label]
            mean = np.mean(group, axis=0)
            means.append(mean)
        self.centroids = means

        return self

    def max_min_values(self):
        minimum = np.min(self.data, axis=0)
        maximum = np.max(self.data, axis=0)
        return maximum,minimum 
    
    def pipeline(self):
        # početni random centroidi
        min_vals, max_vals = self.max_min_values()
        #self.centroids = np.random.uniform(low=min_vals, high=max_vals, size=(self.n_clusters, len(self.data[0])))
        self.centroids = random.sample(list(self.data), self.n_clusters)
        # prva promjena centroida
        old_centroids = self.centroids.copy()
        counter = 0
        while True:
            self.assign_clusters()
            self.update_centroids()
            new_centroids = self.centroids
            counter += 1
            if np.allclose(old_centroids, new_centroids) or counter > 50:
                break
            old_centroids = new_centroids
        
        
        self.centroids = new_centroids
        self.assign_clusters()
        return self.clusters
    
    
    
                    
        
        