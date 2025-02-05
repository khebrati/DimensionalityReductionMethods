import numpy as np

def _compute_distance_matrix(X):
    """
    Compute the pairwise Euclidean distance matrix for the dataset X.
    
    Parameters:
    - X: numpy array of shape (m, n)
    
    Returns:
    - D: numpy array of shape (m, m) where D[i, j] is the Euclidean distance 
         between X[i] and X[j].
    """
    # Compute squared sum for each row
    sum_X = np.sum(X**2, axis=1)
    # Compute squared Euclidean distance using vectorized operations
    sq_dists = sum_X.reshape(-1, 1) + sum_X.reshape(1, -1) - 2 * np.dot(X, X.T)
    # Ensure non-negative (may get small negative values due to numerical errors)
    sq_dists = np.maximum(sq_dists, 0)
    # Take square root to get Euclidean distances
    D = np.sqrt(sq_dists)
    return D


class KNearestNeighbors:
    """
    Compute the k-nearest neighbors for each point in the dataset.
    
    Attributes:
    - k: int, the number of nearest neighbors to find.
    """
    
    def __init__(self, k):
        self.k = k
        
    def __call__(self, X):
        """
        Parameters:
        - X: numpy array, the dataset (m x n).
        
        Returns:
        - neighbors: numpy array, adjacency matrix (m x m) where a 1 indicates a neighbor.
          For each point, the indices corresponding to the k smallest distances are 1, others 0.
        """
        m = X.shape[0]
        D = _compute_distance_matrix(X)
        neighbors = np.zeros((m, m), dtype=int)
        
        # Loop over each point (allowed as an outer loop)
        for i in range(m):
            # Sort distances in ascending order and select the indices of the k smallest distances.
            # Note: This includes the point itself, since its distance is 0.
            idx = np.argsort(D[i])
            k_idx = idx[:self.k]
            neighbors[i, k_idx] = 1
        return neighbors
        

class EpsNeighborhood:
    """
    Compute the epsilon-neighborhood for each point in the dataset.
    
    Attributes:
    - epsilon: float, the maximum distance to consider a point as a neighbor.
    """
    
    def __init__(self, eps):
        self.eps = eps
        
    def __call__(self, X):
        """
        Parameters:
        - X: numpy array, the dataset (m x n).
        
        Returns:
        - neighbors: numpy array, adjacency matrix (m x m) where a 1 indicates a neighbor.
          For each point, any other point within distance eps is considered a neighbor.
        """
        m = X.shape[0]
        D = _compute_distance_matrix(X)
        neighbors = np.zeros((m, m), dtype=int)
        
        # Loop over each point (outer loop is acceptable)
        for i in range(m):
            # Determine which points fall within the epsilon radius.
            neighbor_mask = D[i] <= self.eps
            # Mark these as neighbors (convert boolean to integer)
            neighbors[i, :] = neighbor_mask.astype(int)
        return neighbors
