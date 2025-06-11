import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings("ignore")

class SpectralClusteringKVV:
    def __init__(self, sigma_X, adjustion, lanczos_k=None, l=None, cheeger_cond_max=None, sigma_I=None, r=None):
        
        self.sigma_I = sigma_I #intsnsitiy scale
        self.sigma_X = sigma_X #spatial sclae
        self.r = r # spatial cutoff
        self.lanczos_k = lanczos_k
        self.l = l
        self.cheeger_cond_max = cheeger_cond_max  # threshold za splitting
        self.current_cluster_id = 0
        #self.max_clusters = max_clusters
        #vrijednosti za ispits
        self.all_cheeger_cond_values = []  # za sve cheeger_cond vrijednosti
        self.fiedler_vectors = []
        self.splits = []  # za sve splits
        self.sorted_indexes = []
        self.adjustion = adjustion
        self.L_subs = []
        self.L_subs_indices = []

    #-----------------------učitavanje slike-----------------------------
    def load_image(self, image_path):
        self.img = io.imread(image_path, as_gray=True).astype(np.float64)
        # skaliranje brigthnessa na 0-255
        self.img *= 255  
        self.img = np.clip(self.img, 0, 255)
        self.rows, self.cols = self.img.shape
        #------------------pretvaranje pixela u čvorove------------------
        self.X = np.array([(i, j) for i in range(self.rows) for j in range(self.cols)])
        self.intensities = self.img.flatten()
        self.n = self.X.shape[0]
        self.clusters = np.zeros(self.n, dtype=int) # ???
        return self
    
    """ 1.KORAK """
    #---------------------- iz rada SM formula za simm.----------------------
    def compute_similarity_matrix(self):
        W = np.zeros((self.n, self.n))  # simmilarity matrix prema brightness
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
    #---------------------- eigenvrijednosti - eigenvektori ----------------
    def compute_laplacian(self, W=None):
        #L = D - W -> nenormalizirana matrica
        if W is None:
            W = self.W
        D = np.diag(np.sum(W, axis=1))
        # PROBLEM: nekad naiđe na dijeljenje s 0
        # 
        D_inv_sqrt = np.diag([1.0 / np.sqrt(d) if d != 0 else 0 for d in np.diag(D)]) # D^1/2
        self.L = D_inv_sqrt @ (D - W) @ D_inv_sqrt


    # koristimo Lanczovou metodu da kretiramo manju matricu T umjesto L za 
    # izračunavanje eigen... built-in metodom .eigh
    # https://arxiv.org/pdf/2410.11090
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
            # TODO: callable???
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
        #b = np.random.rand(self.n)
        b = np.ones(L.shape[0]) # da maknemo randomness
        if (self.lanczos_k != None):
            Q, alphas, betas = self.lanczos(L, b)
            # tridijagonalna matrica
            T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        # TODO: zamijeniti built-in .eigh funkciju ??
        if (len(self.X[0]) == 2):
            self.eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]
        else:
            self.eigvals, eigvecs = np.linalg.eigh(T)
            fiedler = Q @ eigvecs[:, 1]
        fiedler = np.sign(fiedler[np.argmax(np.abs(fiedler))]) * fiedler # consistent sign ??
        return fiedler

    """ 3.KORAK """
    #----------------------------- Cheegerova provodljivost ----------------------------------
    def compute_cheeger(self, W, D, A, B):
        cut_AB = np.sum(W[A, :][:, B])
        # vol je assoc
        assoc_A = np.sum(D[A])
        assoc_B = np.sum(D[B])
        return cut_AB / max(min(assoc_A, assoc_B), 1e-12)

    def recursive_two_way(self, indices, parent_cluster_id=0):
        # logika da se i prati max broj grupa: or self.current_cluster_id >= self.max_clusters - 1
        if (self.l):
           if len(indices) < self.l :
                self.clusters[indices] = parent_cluster_id
                return
        elif len(indices) < 2:
            self.clusters[indices] = parent_cluster_id
            return

        # kvv mijenja L matricu
        W_sub = self.W[indices][:, indices]
        self.L_sub = self.L[indices][:, indices]
        self.L_subs_indices.append([self.L_sub, indices])
        #kvv_mult
        if (self.adjustion == "kvv_mult" and len(indices) != self.n):
            L_sub_copy = self.L_sub.copy()
            #skaliram sve elemente da suma retka bude 1
            row_sums = np.sum(self.L_sub, axis=1, keepdims=True)
            # izbjegni dijeljenje s 0
            row_sums[row_sums == 0] = 1
            self.L_sub = self.L_sub / row_sums
            self.L_subs.append([L_sub_copy, row_sums, self.L_sub])  # spremanje L_sub matrice za svaki podgraf
            
        elif (self.adjustion == "kvv_add" and len(indices) != self.n):
            # dodajem elementima na dijagonali da suama retka bude 1
            L_sub_copy = self.L_sub.copy()
            row_sums = np.sum(self.L_sub, axis=1)
            delta = 1.0 - row_sums
            # dodaj na dijagonalu
            self.L_sub[np.diag_indices_from(self.L_sub)] += delta
            self.L_subs.append([L_sub_copy, delta, self.L_sub])  # spremanje L_sub matrice za svaki podgraf
        self.fiedler = self.compute_fiedler_vector(self.L_sub)
        self.fiedler_vectors.append(self.fiedler)  # spremanje fiedlerovog vektora za svaki podgraf
        
        #------------------------ Find optimal split ----------------------------
        self.sorted_idx = np.argsort(self.fiedler)
        self.sorted_indexes.append(self.sorted_idx)  # spremanje indeksa sortiranja fiedlerovog vektora
        min_cheeger_cond = np.inf
        best_split = self.l
        self.cheeger_cond_values = []
                
        # ako je 2d data, onda l mora biti 1
        if (len(self.X[0]) == 2):
            current_cheeger_cond_list = []
            for i in range(1, len(self.fiedler)):
                A = self.sorted_idx[:i]
                B = self.sorted_idx[i:]
                current_cheeger_cond = self.compute_cheeger(W_sub, np.diag(np.sum(W_sub, axis=1)), A, B)
                current_cheeger_cond_list.append(current_cheeger_cond)  # spremanje svih cheeger_cond vrijednosti
                if current_cheeger_cond < min_cheeger_cond:
                    min_cheeger_cond = current_cheeger_cond
                    best_split = i
                    
                    self.cheeger_cond_values.append(current_cheeger_cond) 
            self.all_cheeger_cond_values.append(current_cheeger_cond_list) 
        else:   
            for i in range(self.l, len(self.fiedler) - self.l, self.l):
                A = self.sorted_idx[:i]
                B = self.sorted_idx[i:]
                current_cheeger_cond = self.compute_cheeger(W_sub, np.diag(np.sum(W_sub, axis=1)), A, B)
                self.all_cheeger_cond_values.append(current_cheeger_cond)  # spremanje svih cheeger_cond vrijednosti

                if current_cheeger_cond < min_cheeger_cond:
                    min_cheeger_cond = current_cheeger_cond
                    best_split = i
                    self.cheeger_cond_values.append(current_cheeger_cond) 
        "4. KORAK "
        #----------------------------- cut - stability ----------------------------
        # kako odrediti max cut:
            # empririjski [0.1,0.5] ->provjeriti
            # malo iznad prosjeka
        #self.max_cheeger_cond = np.mean(cheeger_cond_values) # previše grupa
        
        # dijeliti ili ne
        # logika da se prati  i broj grupa: """ and self.current_cluster_id < self.max_clusters - 1 """
        if min_cheeger_cond < self.cheeger_cond_max:  
            self.splits.append({best_split: min_cheeger_cond}) 
            left = indices[self.sorted_idx[:best_split]]
            right = indices[self.sorted_idx[best_split:]]
            
            "5. KORAK"
            #----------------------------- two-way recursion ----------------------------
            self.current_cluster_id += 1
            new_cluster_id = self.current_cluster_id
            
            self.clusters[right] = new_cluster_id
            self.recursive_two_way(left, parent_cluster_id)
            self.recursive_two_way(right, new_cluster_id)
        else:
            self.clusters[indices] = parent_cluster_id
            
       

    # pipline koji poziva sve bitne funkcije
    def segment_image(self):
        self.compute_similarity_matrix()
        self.compute_laplacian()
        self.recursive_two_way(np.arange(self.n))
        return self.clusters.reshape((self.rows, self.cols))
    
    #---------------------------------- average color ---------------------------------------
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
        axs[1].set_title('Spectralno Grupiranje: cheeger_cond + Lanczos')
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    #2d dio
    def load_2d_data(self, data_2d):
        self.X = np.array(data_2d)
        self.intensities = np.zeros(len(data_2d))  
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
    
    def segment_2d(self,data):
        self.load_2d_data(data)
        self.compute_similarity_matrix_2d_gauss()
        self.compute_laplacian()
        self.recursive_two_way(np.arange(self.n))
        return self.clusters.reshape((self.rows, self.cols))

    