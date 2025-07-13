import numpy as np
import importlib
import SMAlgorithm
importlib.reload(SMAlgorithm)
SpectralClusteringSM = SMAlgorithm.SpectralClusteringSM


class SpectralClusteringKVV(SpectralClusteringSM):
    def __init__(self, sigma_X, adjustion, lanczos_k=None, l=None, cheeger_cond_max=None, sigma_I=None, r=None):
        super().__init__(sigma_X=sigma_X, lanczos_k=lanczos_k, l=l, sigma_I=sigma_I, r=r) 
        self.adjustion = adjustion
        self.cheeger_cond_max = cheeger_cond_max

        self.L_subs = []
        self.L_subs_indices = []
        self.splits = []
        self.fiedler_vectors = []
        self.sorted_indexes = []
        self.ncut_values = []
        self.all_cheeger_cond_values = []   

    #------------------------------ C H E E G E R   V A L U E --------------------------------------------------#
    def compute_cheeger(self, A_m, D, A, B):
        cut_AB = np.sum(A_m[A, :][:, B])
        assoc_A = np.sum(D[A])
        assoc_B = np.sum(D[B])
        return cut_AB / max(min(assoc_A, assoc_B), 1e-12)

    #-------------------------------------------------------------------------------#
    def recursive_two_way(self, indices, parent_cluster_id=0):
        if (self.l):
           if len(indices) < self.l :
                self.clusters[indices] = parent_cluster_id
                return
        elif len(indices) < 2:
            self.clusters[indices] = parent_cluster_id
            return

        W_sub = self.A[indices][:, indices]
        self.L_sub = self.L[indices][:, indices]
        self.L_subs_indices.append([self.L_sub, indices])
        if (self.adjustion == "kvv_mult" and len(indices) != self.n):
            L_sub_copy = self.L_sub.copy()
            row_sums = np.sum(self.L_sub, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            self.L_sub = self.L_sub / row_sums
            self.L_subs.append([L_sub_copy, row_sums, self.L_sub]) 
            
        elif (self.adjustion == "kvv_add" and len(indices) != self.n):
            L_sub_copy = self.L_sub.copy()
            row_sums = np.sum(self.L_sub, axis=1)
            delta = 1.0 - row_sums
            self.L_sub[np.diag_indices_from(self.L_sub)] += delta
            self.L_subs.append([L_sub_copy, delta, self.L_sub])  
        self.fiedler = self.compute_fiedler_vector(self.L_sub)
        self.fiedler_vectors.append(self.fiedler) 
        
        self.sorted_idx = np.argsort(self.fiedler)
        self.sorted_indexes.append(self.sorted_idx) 
        min_cheeger_cond = np.inf
        best_split = self.l
        self.cheeger_cond_values = []
                
        if (self.l == None):
            current_cheeger_cond_list = []
            for i in range(1, len(self.fiedler)):
                A = self.sorted_idx[:i]
                B = self.sorted_idx[i:]
                current_cheeger_cond = self.compute_cheeger(W_sub, np.diag(np.sum(W_sub, axis=1)), A, B)
                current_cheeger_cond_list.append(current_cheeger_cond) 
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
                self.all_cheeger_cond_values.append(current_cheeger_cond) 

                if current_cheeger_cond < min_cheeger_cond:
                    min_cheeger_cond = current_cheeger_cond
                    best_split = i
                    self.cheeger_cond_values.append(current_cheeger_cond) 
                    
        if min_cheeger_cond < self.cheeger_cond_max:  
            self.splits.append({best_split: min_cheeger_cond}) 
            left = indices[self.sorted_idx[:best_split]]
            right = indices[self.sorted_idx[best_split:]]
            
            self.current_cluster_id += 1
            new_cluster_id = self.current_cluster_id
            
            self.clusters[right] = new_cluster_id
            self.recursive_two_way(left, parent_cluster_id)
            self.recursive_two_way(right, new_cluster_id)
        else:
            self.clusters[indices] = parent_cluster_id
            
    