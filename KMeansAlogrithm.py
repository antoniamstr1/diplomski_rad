import numpy as np
import random

class KMeansCustom:
    def __init__(self, n_clusters, data):
        self.n_clusters = n_clusters
        self.data = np.array(data)

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # PROBLEM s NaN vrijdnostima: nekada neke grupe ne bi obuhvatile niti jednu točku -> rez je prazna slika
    # implementiran kmeans++
    # http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
    def initialize_centroids(self):
        # uzimam prvu random točku kao početni centroid
        # 1a.
        centroids = [self.data[np.random.randint(0, len(self.data))]]
        # za sljedećih n središta
        for _ in range(1, self.n_clusters):
            # za svaku točku izračunava udaljenost do najbližeg centroida
            distances = np.array([min([self.euclidean_distance(d, c)**2 for c in centroids]) for d in self.data])
            # izračunavam vjerojatnosti da sljedeća točka bude odabrana D(x^2)
            # normaliziramo distribuciju u probabilty distribution
            # 1b.
            probs = distances / distances.sum()
            # pretvara u kumulativnu distribuciju tako da je u [0,1] i možemo ju uzorkovati 
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids.append(self.data[j])
                    break
        return np.array(centroids)

    def assign_clusters(self):
        clusters = []
        for d in self.data:
            # dodajemo index središta do kojeg teba najmanje
            closest_centroid = np.argmin([self.euclidean_distance(d, c) for c in self.centroids])
            clusters.append(closest_centroid)
        self.clusters = np.array(clusters)
        return self

    def update_centroids(self):
        new_centroids = []
        for label in range(self.n_clusters):
            group = self.data[self.clusters == label]
            if len(group) == 0:
                # ako nema grupa točaka, dodajemo novi centroid
                new_centroid = self.data[np.random.randint(0, len(self.data))]
            else:
                new_centroid = np.mean(group, axis=0)
            new_centroids.append(new_centroid)
        self.centroids = np.array(new_centroids)
        return self
    
    def max_min_values(self):
            minimum = np.min(self.data, axis=0)
            maximum = np.max(self.data, axis=0)
            return maximum,minimum 
        
    def pipeline(self):
        # početni random centroidi
        #min_vals, max_vals = self.max_min_values()
        #self.centroids = np.random.uniform(low=min_vals, high=max_vals, size=(self.n_clusters, len(self.data[0])))
        self.centroids = self.initialize_centroids()
        old_centroids = self.centroids.copy()
        while True:
            self.assign_clusters()
            self.update_centroids()
            if np.allclose(old_centroids, self.centroids):
                break
            old_centroids = self.centroids.copy()
        return self.clusters
