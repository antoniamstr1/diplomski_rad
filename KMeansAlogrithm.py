import numpy as np
import random
from skimage import io
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


class KMeansCustom:
    def __init__(self, n_clusters, data=None):
        self.n_clusters = n_clusters
        self.data = np.array(data) if data is not None else np.array([])

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
    
    def initialize_centroids_basic_algorithm(self):
        indices = np.random.choice(len(self.data), self.n_clusters, replace=False)
        centroids = self.data[indices]
        return centroids


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
        
    """ pipeline koji vraća clustere """
    def pipeline(self):
        # početni random centroidi
        #min_vals, max_vals = self.max_min_values()
        #self.centroids = np.random.uniform(low=min_vals, high=max_vals, size=(self.n_clusters, len(self.data[0])))
        self.centroids = self.initialize_centroids()
        old_centroids = self.centroids.copy()

        self.centroid_history = [old_centroids.copy()]   # store initial centroids
        self.cluster_history = []

        self.counter = 50000
        while True and self.counter > 0:
            self.counter -= 1
            self.assign_clusters()
            self.cluster_history.append(self.clusters.copy())  # store current clustering

            self.update_centroids()
            self.centroid_history.append(self.centroids.copy())  # store updated centroids

            if np.allclose(old_centroids, self.centroids):
                break

            old_centroids = self.centroids.copy()
        return self.clusters, self.centroid_history

    
    
    """ SEGMENTACIJA SLIKA """
    def load_image(self, image_path):
        self.data = io.imread(image_path, as_gray=True).astype(np.float64)
        # skaliranje brigthnessa na 0-255
        self.data *= 255  
        self.data = np.clip(self.data, 0, 255)
        self.rows, self.cols = self.data.shape
        #------------------pretvaranje pixela u čvorove-----------------
        self.X = np.array([
        (i, j, self.data[i, j]) for i in range(self.rows) for j in range(self.cols)
        ])
        self.n = self.X.shape[0]
        self.clusters = np.zeros(self.n, dtype=int) # ???
        return self
    #---------------------------------- average color ---------------------------------------
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
        
    """ pipline koji dolazi iz slike """
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
        