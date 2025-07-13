import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.colors as mcolors

from KMeansAlogrithm import KMeansCustom
from scipy.spatial.distance import pdist, squareform


class SpectralClusteringNJW:
    def __init__(self, max_clusters, sigma_X=None, sigma_I=None, r=None):
        self.sigma_I = sigma_I 
        self.sigma_X = sigma_X 
        self.r = r  
        self.max_clusters = max_clusters
        self.all_eigenvecs = []
        self.data = None
        
        self.X = None
        self.img = None
        self.intensities = None
        self.rows = None
        self.cols = None
        self.n = None
        
        self.A = None
        self.L = None
        
        self.eigvals = []
        self.eigvecs = []
        
        self.X = None
        self.Y = None

    #-------------------------------------------------- D A T A  L O A D I N G --------------------------------------------------#
    def load_data(self, data, mode):
        if mode == 'image':
            self.img = io.imread(data, as_gray=True).astype(np.float64)
            self.img *= 255
            self.img = np.clip(self.img, 0, 255)
            self.rows, self.cols = self.img.shape
            self.data = np.array([(i, j) for i in range(self.rows) for j in range(self.cols)])
            self.intensities = self.img.flatten()
            self.n = self.data.shape[0]
        elif mode == 'points':
            self.data = np.array(data)
            self.n = self.data.shape[0]
            self.rows = 1
            self.cols = self.n
        else:
            raise ValueError(f"Unknown mode '{mode}'. Supported modes: 'image', 'None'")

        self.clusters = np.zeros(self.n, dtype=int)
        return self
    
        #-------------------------------------------------- S I M I L A R I T Y  M A T R I X --------------------------------------------------#
    def compute_similarity_matrix(self, mode):
        A = np.zeros((self.n, self.n))

        if mode == 'image':
            r_sq = self.r ** 2
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    spatial_dist_sq = np.linalg.norm(self.data[i] - self.data[j])
                    if spatial_dist_sq < r_sq:
                        intensity_diff_sq = np.linalg.norm(self.intensities[i] - self.intensities[j]) ** 2
                        w_ij = np.exp(-intensity_diff_sq / (self.sigma_I ** 2)) * \
                            np.exp(-spatial_dist_sq**2 / (self.sigma_X ** 2))
                        A[i, j] = w_ij
                        A[j, i] = w_ij

        elif mode == 'gauss':
            A = np.zeros((self.n, self.n))
            distances = squareform(pdist(self.data))
            A = np.exp(-(distances**2) / (2 * self.sigma_X**2))
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    dist_sq = np.linalg.norm(self.data[i] - self.data[j]) ** 2
                    w_ij = np.exp(-dist_sq / (2 * self.sigma_X**2))
                    A[i, j] = w_ij
                    A[j, i] = w_ij
                    
        elif mode == 'cosine':
            distances = squareform(pdist(self.data, metric='cosine'))
            A = 1 - distances

        else:
            raise ValueError(f"Unknown mode '{mode}'. Supported modes: 'rgb+xy', '2d-gauss', 'nd-cosine'.")

        np.fill_diagonal(A, 0)
        self.A = A
        return self
    
    
    #------------------------------N O R M A L I Z E D    L A P L A C I A N --------------------------------------------------#
    def compute_laplacian(self, A=None):
        if A is None:
            A = self.A
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.diag([1.0 / np.sqrt(d) if d != 0 else 0 for d in np.diag(D)]) 
        self.L = D_inv_sqrt @ (D - A) @ D_inv_sqrt
        return self.L
    

    #------------------------------  E I G E N V E C T O R S --------------------------------------------------#
    def compute_k_eigenvectors(self):
        L = self.compute_laplacian(self.A)

        self.eigvals, self.eigvecs = np.linalg.eigh(L)
        self.X = self.eigvecs[:, : self.max_clusters]
        self.Y = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)

        return self.Y

    def segment_image(self):
        self.compute_similarity_matrix()
        Z = self.compute_k_eigenvectors()
        customKmeans = KMeansCustom(self.max_clusters, Z)
        self.clusters, _ = customKmeans.pipeline()
        return np.array(self.clusters).reshape((self.rows, self.cols))

    #-------------------------------------------------- S E G M E T A T I O N --------------------------------------------------#
    def segment(self, data, mode, similarity_type):
        self.load_data(data, mode=mode)
        self.compute_similarity_matrix(mode=similarity_type)
        Z = self.compute_k_eigenvectors()
        customKmeans = KMeansCustom(self.max_clusters, Z)
        self.clusters, _ = customKmeans.pipeline()
        return np.array(self.clusters).reshape((self.rows, self.cols))
    
    #-------------------------------------------------- A V E R A G E  C O L O R --------------------------------------------------#
    def average_color(self):
        self.segmented_img = np.array(self.clusters).reshape((self.rows, self.cols))

        group_sums = np.zeros(self.max_clusters)
        group_counts = np.zeros(self.max_clusters)

        for i in range(self.segmented_img.shape[0]):
            for j in range(self.segmented_img.shape[1]):
                group_label = self.segmented_img[i, j]
                group_value = self.img[i, j]

                group_sums[group_label] += group_value
                group_counts[group_label] += 1

        group_averages = (group_sums / group_counts) / 255
        rgb_grouped = [
            (group_averages[i], group_averages[i], group_averages[i])
            for i in range(0, len(group_averages))
        ]
        cmap = mcolors.ListedColormap(rgb_grouped)

        return cmap

    #-------------------------------------------------- 3 D   V I S U A L I Z A T I O N --------------------------------------------------#
    def visualize_3d_clusters(self):
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(self.Y[:, 0], self.Y[:, 1], self.Y[:, 2], c=self.clusters, cmap="viridis", s=50)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend(*scatter.legend_elements(), title="Skupine", loc="upper right")
        plt.tight_layout()
        plt.show()
