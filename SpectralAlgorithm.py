import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform



class SpectralClustering:
    def __init__(self, X=None, k=5, sigma=None):
        self.X = X                     
        self.k = k                      
        self.sigma = sigma        
        self.A = None               
        self.D = None             
        self.L = None        
        self.L_norm = None      
        self.eigvals = None           
        self.eigvecs = None             
        self.fiedler_vector = None    
        self.clusters = None           
        
    def compute_adjacency_matrix(self):
        n = len(self.X)
        A = np.ones((n, n)).astype(int)
        np.fill_diagonal(A, 0)
        self.A = A
        return A
    
    def compute_similarity_matrix_euclidian(self):
        n = len(self.X)
        D = cdist(self.X, self.X, metric='euclidean')
        A = np.zeros_like(D)
        for i in range(n):
            for j in range(n):
                if i != j:
                    A[i, j] = 1 / D[i, j]
        self.A = A
        return self.A
    
    def compute_knn_similarity_matrix(self, k):
        self.compute_knn_adjacency_matrix(k)
        D = cdist(self.X, self.X, metric='euclidean')
        S = np.zeros_like(D)
        n = len(self.X)
        for i in range(n):
            for j in range(n):
                if i != j:
                    S[i, j] = 1 / D[i, j]
        self.A = S * self.A
        return self.A
    
    def compute_knn_adjacency_matrix(self, k):
        n = len(self.X)
        A = np.ones((n, n)).astype(int)
        self.A = np.zeros_like(A)
        D = cdist(self.X, self.X, metric='euclidean')
        for i in range(n):
            knn_indices = np.argsort(D[i])[1:k+1]
            self.A[i, knn_indices] = 1
        return self.A
            
    def compute_similarity_matrix(self):
        n = len(self.X)
        A = np.zeros((n, n))
        distances = squareform(pdist(self.X))
        A = np.exp(-distances**2 / (2 * self.sigma**2))
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.linalg.norm(self.X[i] - self.X[j]) ** 2
                w_ij = np.exp(-dist_sq / (2 * self.sigma ** 2))
                A[i, j] = w_ij
                A[j, i] = w_ij
        np.fill_diagonal(A, 0)
        self.A = A
        return A

    def compute_degree_matrix(self, A=None):
        """ Izračun matrice stupnjeva """
        if A is None:
            A = self.A
        D = np.diag(np.sum(A, axis=1))
        self.D = D
        return D

    def compute_laplacian(self, A=None):
        """ Izračun Laplaceove matrice L = D - A """
        if A is None:
            A = self.A
        if self.D is None:
            self.compute_degree_matrix(A)
        L = self.D - A
        self.L = L
        return L

    def compute_normalized_laplacian(self, A=None):
        """ Izračun simetrično normalizirane Laplaceove matrice L_sym = D^(-1/2) L D^(-1/2) """
        if A is None:
            A = self.A
        if self.L is None:
            self.compute_laplacian(A)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(self.D) + 1e-10))  # Stabilnost
        L_norm = D_inv_sqrt @ self.L @ D_inv_sqrt
        self.L_norm = L_norm
        return L_norm

    def compute_fiedler_vector(self, use_normalized=True):
        """ Izračun Fiedlerovog vektora i podjela u 2 klastera """
        L = self.L_norm if use_normalized else self.L
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.fiedler_vector = eigvecs[:, 1]
        self.clusters = np.where(self.fiedler_vector >= 0, 1, 0)
        return self.fiedler_vector, self.clusters

    def pipeline(self, normalized=True, sigma=None):
        if sigma is not None:
            self.sigma = sigma
        """ Pokreni cijeli postupak spektralnog klasteriranja """
        self.compute_similarity_matrix()
        self.compute_degree_matrix()
        self.compute_laplacian()
        if normalized:
            self.compute_normalized_laplacian()
        self.compute_fiedler_vector(use_normalized=normalized)
        return self.clusters

    def create_graph(self):
        G = nx.Graph()
        n = self.X.shape[0]
        for i in range(n):
            G.add_node(i, pos=self.X[i])
        for i in range(n):
            for j in range(i + 1, n):
                # dodavanje bridova samo ako imaju težinu """
                if self.A[i, j] > 0.01:
                    G.add_edge(i, j, weight=self.A[i, j])
                    
        edges = G.edges(data=True)
        weights = [edge[2]['weight'] for edge in edges]
        min_width, max_width = 1, 5
        min_w, max_w = min(weights), max(weights)
        self.edge_widths = [min_width + (w - min_w) / (max_w - min_w)
                    * (max_width - min_width) for w in weights]
        pos = nx.get_node_attributes(G, 'pos')
        plt.figure(figsize=(5, 2))
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                edge_color='gray', width=self.edge_widths, node_size=500)
        plt.title("Graf s bridovima iz matrice A")
        plt.axis('equal')
        plt.show()