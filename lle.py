import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
from dataset import load_dataset,visualize_plane
from geo import KNearestNeighbors

class LLE:
    """
    Locally Linear Embedding for nonlinear dimensionality reduction.
    """
    
    def __init__(self, n_components, *, adj_calculator=KNearestNeighbors(5)):
        """
        Initialize LLE with the number of components and neighbors.

        Parameters:
        - n_components: int, the number of dimensions to retain in the reduced space.
        - adj_calculator: function, given a dataset, returns the adjacency matrix.
        """
        self.n_components = n_components
        self._adj_calculator = adj_calculator
        
    def _compute_weights(self, X, distances=None):
        """
        Compute weights for each point using a constrained least squares problem.
        
        For each data point xi, we solve:
            min_w || xi - sum_{j in N(i)} w_j * xj ||^2 
        subject to sum_j w_j = 1.
        
        This is solved by:
            1. Forming the local difference matrix: Z = xi - X[N(i)]
            2. Computing the local covariance matrix: C = Z * Z^T
            3. Regularizing C:  C_reg = C + reg * trace(C) * I (to ensure stability)
            4. Solving the linear system: C_reg * w = 1 (a vector of ones)
            5. Normalizing w so that sum(w) = 1.
        
        Parameters:
        - X: numpy array, shape (m, n)
        
        Returns:
        - W: numpy array, shape (m, m) where W[i, j] is the weight of neighbor j for point i.
          Only k-nearest neighbors (as determined by the adjacency matrix from _adj_calculator) have nonzero weights.
        """
        m = X.shape[0]
        adj = self._adj_calculator(X)
        W = np.zeros((m, m))
        reg = 1e-3

        for i in range(m):
            neighbors = np.where(adj[i] == 1)[0]
            if neighbors.size == 0:
               continue
            
            Xi = X[i]
            Z = X[neighbors] - Xi
            
            C = np.dot(Z, Z.T)
            C = C + np.eye(C.shape[0]) * reg * np.trace(C)
            
            ones = np.ones(neighbors.shape[0])
            try:
                w = np.linalg.solve(C, ones)
            except np.linalg.LinAlgError:
                w = np.dot(np.linalg.pinv(C), ones)
            
            w = w / np.sum(w)
            W[i, neighbors] = w
            
        return W

    def _compute_embedding(self, W):
        """
        Compute the embedding Y using the eigenvectors corresponding to the smallest
        nonzero eigenvalues of the matrix M = (I - W)^T (I - W).

        Parameters:
        - W: numpy array, shape (m, m)
        
        Returns:
        - Y: numpy array, shape (m, n_components) containing the low-dimensional embedding.
        """
        m = W.shape[0]
        I = np.eye(m)
        M = np.dot((I - W).T, (I - W))
        eigvals, eigvecs = np.linalg.eigh(M)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        Y = eigvecs[:, 1:self.n_components+1]
        return Y

    def fit_transform(self, X):
        """
        Fit the LLE model to the dataset and reduce its dimensionality.

        Parameters:
        - X: numpy array, the dataset (m x n).

        Returns:
        - Y: numpy array, shape (m, n_components) containing the low-dimensional embeddings.
        """
        W = self._compute_weights(X)
        Y = self._compute_embedding(W)
        return Y

if __name__ == "__main__":
    X, labels = load_dataset("datasets/swissroll.npz")
    lle_model = LLE(n_components=2,adj_calculator=KNearestNeighbors(15))
    # lle_model = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
    Y = lle_model.fit_transform(X)
    visualize_plane(Y, labels)
    print("Low-dimensional embedding shape:", Y.shape)
