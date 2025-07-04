import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings("ignore")

class SpectralClusteringSM:
    def __init__(self, sigma_X, lanczos_k=None, l=None, ncut_max=None, sigma_I=None, r=None):
        
        self.sigma_I = sigma_I 
        self.sigma_X = sigma_X 
        self.r = r
        self.lanczos_k = lanczos_k
        self.l = l
        self.ncut_max = ncut_max  
        self.current_cluster_id = 0
        self.all_ncut_values = []  
        self.fiedler_vectors = []
        self.splits = [] 
        self.sorted_indexes = []
        self.all_eigenvals = []

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
        r_sq = self.r ** 2
        for i in range(self.n):
                for j in range(i + 1, self.n):
                    spatial_dist_sq = np.linalg.norm(self.X[i] - self.X[j])
                    if spatial_dist_sq < r_sq:
                        intensity_diff_sq = np.linalg.norm(self.intensities[i] - self.intensities[j]) ** 2
                        w_ij = np.exp(-intensity_diff_sq / (self.sigma_I ** 2)) * \
                            np.exp(-spatial_dist_sq**2 / (self.sigma_X ** 2))
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
        self.L = D_inv_sqrt @ (D - W) @ D_inv_sqrt
        return self.L


    def lanczos(self, A, b):
        Q = np.zeros((len(b), self.lanczos_k))
        alphas = np.zeros(self.lanczos_k)
        betas = np.zeros(self.lanczos_k-1)
        
        q_prev = np.zeros(len(b)) # q_-1
        q_curr = b / np.sqrt(np.sum(b**2)) # q_0
        beta_prev = 0.0 # beta_-1
        
        #for 𝑛 = 0, 1, … , 𝑘 − 1 do...
        for i in range(self.lanczos_k):
            Q[:, i] = q_curr
            
            # korak 4: 𝐲𝑛+1 = 𝐀𝐪𝑛 − 𝛽𝑛−1𝐪𝑛−1
            if callable(A):
                y = A*q_curr - beta_prev * q_prev
            else:
                y = A @ q_curr - beta_prev * q_prev
            
            # korak 5: 𝛼𝑛 = 𝐪⊺𝑛*𝐲𝑛+1
            alpha_i = q_curr.T @ y
            alphas[i] = alpha_i
            
            # korak 6: 𝐳𝑛+1 = 𝐲𝑛+1 − 𝛼𝑛𝐪𝑛
            z = y - alpha_i * q_curr
            
            #korak 7: preskok
            if i < self.lanczos_k - 1:
                
                 # korak 8: 𝛽𝑛 = ‖𝐳𝑛+1‖2
                beta_i = np.sqrt(np.sum(z**2))
                betas[i] = beta_i
                
                # korak 9: 𝐪𝑛+1 = 𝐳𝑛+1/𝛽𝑛
                q_prev = q_curr
                q_curr = z / beta_i
                beta_prev = beta_i
        
        return Q, alphas, betas
    

    def compute_fiedler_vector(self, L):
        b = np.ones(L.shape[0]) 
        if (self.lanczos_k != None):
            Q, alphas, betas = self.lanczos(L, b)
            T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        if (len(self.X[0]) == 2 or len(self.X[0]) == 784):
            self.eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]

        else:
            self.eigvals, eigvecs = np.linalg.eigh(T)
            fiedler = Q @ eigvecs[:, 1]
        self.all_eigenvals.append(self.eigvals)
        return fiedler

    """ 3.KORAK """
    def compute_ncut(self, W, D, A, B):
        cut_AB = np.sum(W[A, :][:, B])
        assoc_A = np.sum(D[A])
        assoc_B = np.sum(D[B])
        return (cut_AB / assoc_A) + (cut_AB / assoc_B)

    def recursive_two_way(self, indices, parent_cluster_id=0):
        if (self.l):
           if len(indices) < self.l :
                self.clusters[indices] = parent_cluster_id
                return
        elif len(indices) < 2:
            self.clusters[indices] = parent_cluster_id
            return
        W_sub = self.W[indices][:, indices]
        L_sub = self.compute_laplacian(W_sub)
        self.fiedler = self.compute_fiedler_vector(L_sub)
        self.fiedler_vectors.append(self.fiedler)  
        
        #------------------------ optimalni rez ----------------------------
        self.sorted_idx = np.argsort(self.fiedler)
        self.sorted_indexes.append(self.sorted_idx)  
        min_ncut = np.inf
        best_split = self.l
        self.ncut_values = []
                
        if (len(self.X[0]) == 2 or len(self.X[0]) == 784):
            current_ncut_list = []
            for i in range(1, len(self.fiedler)):
                A = self.sorted_idx[:i]
                B = self.sorted_idx[i:]
                current_ncut = self.compute_ncut(W_sub, np.diag(np.sum(W_sub, axis=1)), A, B)
                current_ncut_list.append(current_ncut)  
                if current_ncut < min_ncut:
                    min_ncut = current_ncut
                    best_split = i
                    
                    self.ncut_values.append(current_ncut) 
            self.all_ncut_values.append(current_ncut_list) 
        else:   
            for i in range(self.l, len(self.fiedler) - self.l, self.l):
                A = self.sorted_idx[:i]
                B = self.sorted_idx[i:]
                current_ncut = self.compute_ncut(W_sub, np.diag(np.sum(W_sub, axis=1)), A, B)
                self.all_ncut_values.append(current_ncut)  

                if current_ncut < min_ncut:
                    min_ncut = current_ncut
                    best_split = i
                    self.ncut_values.append(current_ncut) 
                    
        "4. KORAK "
        if min_ncut < self.ncut_max:  
            self.splits.append({best_split: min_ncut})
            left = indices[self.sorted_idx[:best_split]]
            right = indices[self.sorted_idx[best_split:]]
            
            "5. KORAK"
            self.current_cluster_id += 1
            new_cluster_id = self.current_cluster_id
            
            self.clusters[right] = new_cluster_id
            self.recursive_two_way(left, parent_cluster_id)
            self.recursive_two_way(right, new_cluster_id)
        else:
            self.clusters[indices] = parent_cluster_id
            
       
    def segment_image(self):
        self.compute_similarity_matrix()
        self.recursive_two_way(np.arange(self.n))
        return self.clusters.reshape((self.rows, self.cols))
    
    def average_color(self):
        self.segmented_img = self.clusters.reshape((self.rows, self.cols))
        num_clusters = len(set(self.clusters))
        group_sums = np.zeros(num_clusters)
        group_counts = np.zeros(num_clusters)

        for i in range(self.segmented_img.shape[0]):
            for j in range(self.segmented_img.shape[1]):
                group_label = self.segmented_img[i, j]
                group_value = self.img[i, j]
                
                group_sums[group_label] += group_value
                group_counts[group_label] += 1

        group_averages = (group_sums / group_counts) / 255
        rgb_grouped = [(group_averages[i], group_averages[i], group_averages[i]) for i in range(0, len(group_averages))]
        cmap = mcolors.ListedColormap(rgb_grouped)
    
        return cmap
    

    def visualize(self):
        segmented_img = self.clusters.reshape((self.rows, self.cols))
        cmap_custom = self.average_color()
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(self.img, cmap='gray')
        axs[0].set_title('Original')
        axs[0].axis('off')
        
        axs[1].imshow(segmented_img, cmap=cmap_custom)
        axs[1].set_title('Spectralno Grupiranje: Ncut + Lanczos')
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def load_2d_data(self, data_2d):
        self.X = np.array(data_2d)
        self.intensities = np.zeros(len(data_2d))  
        self.n = self.X.shape[0]
        self.rows, self.cols = 1, self.n
        self.clusters = np.zeros(self.n, dtype=int)
        return self
    
    def load_nd_data(self, data_nd):
        self.X = np.array(data_nd)  
        self.n = self.X.shape[0]  
        self.rows, self.cols = 1, self.n  
        self.clusters = np.zeros(self.n, dtype=int)
        return self
    
    def compute_similarity_matrix_2d_gauss(self):
        W = np.zeros((self.n, self.n))
        distances = squareform(pdist(self.X))
        A = np.exp(-distances**2 / (2 * self.sigma_X**2))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist_sq = np.linalg.norm(self.X[i] - self.X[j]) ** 2
                w_ij = np.exp(-dist_sq / (2 * self.sigma_X ** 2))
                W[i, j] = w_ij
                W[j, i] = w_ij
        np.fill_diagonal(W, 0)
        self.W = W
        return self
    
    
    def compute_similarity_matrix_nd_cosine(self):
        distances = squareform(pdist(self.X, metric='cosine'))
        W = 1 - distances
        np.fill_diagonal(W, 0)
        self.W = W
        return self
    
    def compute_similarity_matrix_nd_gauss(self):
        distances_sq = squareform(pdist(self.X, metric='sqeuclidean'))
        W = np.exp(-distances_sq / (2 * self.sigma_X ** 2))
        np.fill_diagonal(W, 0)
        self.W = W
        return self
    
    def segment_2d(self,data):
        self.load_2d_data(data)
        self.compute_similarity_matrix_2d_gauss()
        self.recursive_two_way(np.arange(self.n))
        return self.clusters.reshape((self.rows, self.cols))

    def segment_nd(self,data, similarity_type):
        self.load_nd_data(data)
        if similarity_type == 'gauss':
            self.compute_similarity_matrix_nd_gauss()
        elif similarity_type == 'cosine':
            self.compute_similarity_matrix_nd_cosine()
        self.recursive_two_way(np.arange(self.n))
        return self.clusters.reshape((self.rows, self.cols))
    