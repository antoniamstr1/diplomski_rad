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
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

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
    
    def initialize_centroids_basic_algorithm(self):
        indices = np.random.choice(len(self.data), self.n_clusters, replace=False)
        centroids = self.data[indices]
        return centroids


    def assign_clusters(self):
        clusters = []
        for d in self.data:
            closest_centroid = np.argmin([self.euclidean_distance(d, c) for c in self.centroids])
            clusters.append(closest_centroid)
        self.clusters = np.array(clusters)
        return self

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
    
    def max_min_values(self):
            minimum = np.min(self.data, axis=0)
            maximum = np.max(self.data, axis=0)
            return maximum,minimum 
        
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

    
    
    """ SEGMENTACIJA SLIKA """
    def load_image(self, image_path):
        self.data = io.imread(image_path, as_gray=True).astype(np.float64)
        self.data *= 255  
        self.data = np.clip(self.data, 0, 255)
        self.rows, self.cols = self.data.shape
        self.X = np.array([
        (i, j, self.data[i, j]) for i in range(self.rows) for j in range(self.cols)
        ])
        self.n = self.X.shape[0]
        self.clusters = np.zeros(self.n, dtype=int) # ???
        return self
    def average_color(self):
        self.segmented_img = np.array(self.clusters).reshape((self.rows, self.cols))

        group_sums = np.zeros(self.n_clusters)
        group_counts = np.zeros(self.n_clusters)

        for i in range(self.segmented_img.shape[0]):
            for j in range(self.segmented_img.shape[1]):
                group_label = self.segmented_img[i, j]
                group_value = self.data[i, j]
                
                group_sums[group_label] += group_value
                group_counts[group_label] += 1

        group_averages = (group_sums / group_counts) / 255
        rgb_grouped = [(group_averages[i], group_averages[i], group_averages[i]) for i in range(0, len(group_averages))]
        cmap = mcolors.ListedColormap(rgb_grouped)
    
        return cmap

        

    def visualize(self):
        self.segmented_img = np.array(self.clusters).reshape((self.rows, self.cols))
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        cmap_custom = self.average_color()
        axs[0].imshow(self.data, cmap='gray')
        axs[0].set_title('Original')
        axs[0].axis('off')
        axs[1].imshow(self.segmented_img, cmap=cmap_custom)
        axs[1].set_title('K-Means grupiranje')
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def pipeline_img(self, image_path):
        self.load_image(image_path)
        kmeans = KMeansCustom(self.n_clusters, self.X)
        self.clusters = kmeans.pipeline()
        self.segmented_img = np.array(self.clusters).reshape((self.rows, self.cols))
        self.visualize()
        return self
    
    
    def pipeline_img_line(self, image_path):
        self.load_image(image_path)
        kmeans = KMeansCustom(self.n_clusters, self.X)
        self.clusters = kmeans.pipeline()
        self.segmented_img = np.array(self.clusters).reshape((self.rows, self.cols))
        return self.segmented_img
        
    def sse(self):
        sse = 0.0
        for i in range(len(self.data)):
            center = self.centroids[self.clusters[i]]
            sse += np.sum((self.data[i] - center) ** 2)
        return sse