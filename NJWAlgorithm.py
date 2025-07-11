import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.colors as mcolors

from KMeansAlogrithm import KMeansCustom
from scipy.spatial.distance import pdist, squareform


class SpectralClusteringNJW:
    def __init__(self, sigma_X, max_clusters, sigma_I=None, r=None):
        self.sigma_I = sigma_I 
        self.sigma_X = sigma_X 
        self.r = r  
        self.max_clusters = max_clusters
        self.all_eigenvecs = []

    def load_image(self, image_path):
        self.img = io.imread(image_path, as_gray=True).astype(np.float64)
        self.img *= 255
        self.img = np.clip(self.img, 0, 255)
        self.rows, self.cols = self.img.shape
        self.X = np.array([(i, j) for i in range(self.rows) for j in range(self.cols)])
        self.intensities = self.img.flatten()
        self.n = self.X.shape[0]
        self.clusters = np.zeros(self.n, dtype=int) 
        return self

    """ 1.KORAK """
    def compute_similarity_matrix(self):
        W = np.zeros((self.n, self.n)) 
        r_sq = self.r**2
        for i in range(self.n):
            for j in range(i + 1, self.n):
                spatial_dist_sq = np.linalg.norm(self.X[i] - self.X[j])
                if spatial_dist_sq < r_sq:
                    if self.intensities is not None:
                        intensity_diff_sq = (
                            np.linalg.norm(self.intensities[i] - self.intensities[j])
                            ** 2
                        )
                    else:
                        intensity_diff_sq = 0 
                    w_ij = np.exp(-intensity_diff_sq / (self.sigma_I**2)) * np.exp(
                        -(spatial_dist_sq**2) / (self.sigma_X**2)
                    )
                    W[i, j] = w_ij
                    W[j, i] = w_ij

        self.W = W
        return self

    """ 2.KORAK """
    def compute_laplacian(self, W=None):
        if W is None:
            W = self.W
        D = np.diag(np.sum(W, axis=1))
        D_inv_sqrt = np.diag([1.0 / np.sqrt(d) if d != 0 else 0 for d in np.diag(D)]) 
        self.Laplacian = D_inv_sqrt @ (D - W) @ D_inv_sqrt
        #return D_inv_sqrt @ (D-w)) @ D_inv_sqrt
        return D_inv_sqrt @ W @ D_inv_sqrt

    def compute_k_eigenvectors(self):
        L = self.compute_laplacian(self.W)

        self.eigvals, self.eigvecs = np.linalg.eigh(L)
        self.all_eigenvecs.append(self.eigvecs)
        """ 3/4.KORAK """
        #self.X = self.eigvecs[:, : self.max_clusters]
        idx = np.argsort(self.eigvals)[::-1]  # descending
        eigvecs = self.eigvecs[:, idx]
        self.X = eigvecs[:, :self.max_clusters]
        self.Y = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)

        return self.Y

    def segment_image(self):
        self.compute_similarity_matrix()
        """ 5.KORAK """
        Z = self.compute_k_eigenvectors()
        customKmeans = KMeansCustom(self.max_clusters, Z)
        self.clusters, _ = customKmeans.pipeline()
        return np.array(self.clusters).reshape((self.rows, self.cols))

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

    def visualize(self):
        self.segmented_img = np.array(self.clusters).reshape((self.rows, self.cols))
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        cmap_custom = self.average_color()
        axs[0].imshow(self.img, cmap="gray")
        axs[0].set_title("Original")
        axs[0].axis("off")
        axs[1].imshow(self.segmented_img, cmap=cmap_custom)
        axs[1].set_title("Spectralno Grupiranje NJW")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()

    def load_2d_data(self, data_2d):
        self.X = np.array(data_2d)
        self.n = self.X.shape[0]
        self.rows, self.cols = 1, self.n
        return self

    def compute_similarity_matrix_2d_gauss(self):
        W = np.zeros((self.n, self.n))
        distances = squareform(pdist(self.X))
        A = np.exp(-(distances**2) / (2 * self.sigma_X**2))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist_sq = np.linalg.norm(self.X[i] - self.X[j]) ** 2
                w_ij = np.exp(-dist_sq / (2 * self.sigma_X**2))
                W[i, j] = w_ij
                W[j, i] = w_ij
        np.fill_diagonal(W, 0)
        self.W = W
        return self

    def segment_2d(self, data):
        self.load_2d_data(data)
        self.compute_similarity_matrix_2d_gauss()
        Z = self.compute_k_eigenvectors()
        customKmeans = KMeansCustom(self.max_clusters, Z)
        self.clusters, _ = customKmeans.pipeline()
        return np.array(self.clusters).reshape((self.rows, self.cols))

    def compute_similarity_matrix_nd_gauss(self):
        distances = squareform(pdist(self.X))
        W = np.exp(-(distances**2) / (2 * self.sigma_X**2))
        np.fill_diagonal(W, 0)
        self.W = W
        return self

    def compute_similarity_matrix_nd_cosine(self):
        distances = squareform(pdist(self.X, metric="cosine"))
        W = 1 - distances
        np.fill_diagonal(W, 0)
        self.W = W
        return self

    def segment_3d(self, data, cosine = None):
        self.load_2d_data(data)
        if cosine is not None:
            self.compute_similarity_matrix_nd_cosine()
        else:
            self.compute_similarity_matrix_nd_gauss()
        Z = self.compute_k_eigenvectors()
        customKmeans = KMeansCustom(self.max_clusters, Z)
        self.clusters, _ = customKmeans.pipeline()
        print()
        return np.array(self.clusters).reshape((self.rows, self.cols))

    def visualize_3d_clusters(self):

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            self.X[:, 0],
            self.X[:, 1],
            self.X[:, 2],
            c=self.clusters,
            cmap="viridis",
            s=50,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend(*scatter.legend_elements(), title="Skupine", loc="upper right")
        plt.tight_layout()
        plt.show()
