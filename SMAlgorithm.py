import numpy as np
from skimage import io
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors

class SpectralClusteringSM:
    def __init__(self, sigma_X=None, lanczos_k=None, l=None, ncut_max=None, sigma_I=None, r=None):
        
        self.sigma_X = sigma_X
        self.lanczos_k = lanczos_k
        self.l = l
        self.ncut_max = ncut_max
        self.sigma_I = sigma_I
        self.r = r

        self.X = None
        self.img = None
        self.intensities = None
        self.rows = None
        self.cols = None
        self.n = None

        self.clusters = None
        self.current_cluster_id = 0

        self.A = None
        self.L = None

        self.eigvals = None
        self.fiedler = None
        self.all_eigenvals = []
        self.fiedler_vectors = []

        self.splits = []
        self.sorted_idx = None
        self.sorted_indexes = []

        self.ncut_values = []
        self.all_ncut_values = []

        self.segmented_img = None
        
    #-------------------------------------------------- D A T A  L O A D I N G --------------------------------------------------#
    def load_data(self, data, mode):
        if mode == 'image':
            self.img = io.imread(data, as_gray=True).astype(np.float64)
            self.img *= 255
            self.img = np.clip(self.img, 0, 255)
            self.rows, self.cols = self.img.shape
            self.X = np.array([(i, j) for i in range(self.rows) for j in range(self.cols)])
            self.intensities = self.img.flatten()
            self.n = self.X.shape[0]
        elif mode == 'points':
            self.X = np.array(data)
            self.n = self.X.shape[0]
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
                    spatial_dist_sq = np.linalg.norm(self.X[i] - self.X[j])
                    if spatial_dist_sq < r_sq:
                        intensity_diff_sq = np.linalg.norm(self.intensities[i] - self.intensities[j]) ** 2
                        w_ij = np.exp(-intensity_diff_sq / (self.sigma_I ** 2)) * \
                            np.exp(-spatial_dist_sq**2 / (self.sigma_X ** 2))
                        A[i, j] = w_ij
                        A[j, i] = w_ij

        elif mode == 'gauss':
            distances = squareform(pdist(self.X))
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    dist_sq = distances[i, j] ** 2
                    w_ij = np.exp(-dist_sq / (2 * self.sigma_X ** 2))
                    A[i, j] = w_ij
                    A[j, i] = w_ij

        elif mode == 'cosine':
            distances = squareform(pdist(self.X, metric='cosine'))
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

    #------------------------------ L A N C Z O S   A L G O R I T H M --------------------------------------------------#
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

    #------------------------------ F I E D L E R ' S   V E C T O R --------------------------------------------------#
    def compute_fiedler_vector(self, L):
        if (self.lanczos_k != None):
            b = np.ones(L.shape[0]) 
            Q, alphas, betas = self.lanczos(L, b)
            T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
            self.eigvals, eigvecs = np.linalg.eigh(T)
            fiedler = Q @ eigvecs[:, 1]
        
        else:
            self.eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]
            
        self.all_eigenvals.append(self.eigvals)
        return fiedler

    #------------------------------ N - C U T   V A L U E --------------------------------------------------#
    def compute_ncut(self, A_m, D, A, B):
        cut_AB = np.sum(A_m[A, :][:, B])
        assoc_A = np.sum(D[A])
        assoc_B = np.sum(D[B])
        return (cut_AB / assoc_A) + (cut_AB / assoc_B)
    
    #-------------------------------------------------------------------------------#
    def recursive_two_way(self, indices, parent_cluster_id=0):
        if (self.l):
           if len(indices) < self.l :
                self.clusters[indices] = parent_cluster_id
                return
        elif len(indices) < 2:
            self.clusters[indices] = parent_cluster_id
            return
        A_sub = self.A[indices][:, indices]
        L_sub = self.compute_laplacian(A_sub)
        self.fiedler = self.compute_fiedler_vector(L_sub)
        self.fiedler_vectors.append(self.fiedler)  
        
        self.sorted_idx = np.argsort(self.fiedler)
        self.sorted_indexes.append(self.sorted_idx)  
        min_ncut = np.inf
        best_split = self.l
        self.ncut_values = []
                
        if self.l is None:
            current_ncut_list = []
            for i in range(1, len(self.fiedler)):
                A = self.sorted_idx[:i]
                B = self.sorted_idx[i:]
                current_ncut = self.compute_ncut(A_sub, np.diag(np.sum(A_sub, axis=1)), A, B)
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
                current_ncut = self.compute_ncut(A_sub, np.diag(np.sum(A_sub, axis=1)), A, B)
                self.all_ncut_values.append(current_ncut)  

                if current_ncut < min_ncut:
                    min_ncut = current_ncut
                    best_split = i
                    self.ncut_values.append(current_ncut) 
                    
        if min_ncut < self.ncut_max:  
            self.splits.append({best_split: min_ncut})
            left = indices[self.sorted_idx[:best_split]]
            right = indices[self.sorted_idx[best_split:]]
            
            self.current_cluster_id += 1
            new_cluster_id = self.current_cluster_id
            
            self.clusters[right] = new_cluster_id
            self.recursive_two_way(left, parent_cluster_id)
            self.recursive_two_way(right, new_cluster_id)
        else:
            self.clusters[indices] = parent_cluster_id
     
    #-------------------------------------------------- S E G M E T A T I O N --------------------------------------------------#
    def segment(self, data, mode, similarity_type):
        self.load_data(data, mode=mode)
        self.compute_similarity_matrix(mode=similarity_type)
        self.compute_laplacian()
        self.recursive_two_way(np.arange(self.n))
        return self.clusters.reshape((self.rows, self.cols))

    #-------------------------------------------------- A V E R A G E  C O L O R --------------------------------------------------#
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
