import numpy as np


def _compute_distance_matrix(X):
    # TODO: Compute pairwise Euclidean distance matrix for X
    return None


class KNearestNeighbors:
    """
    Compute the k-nearest neighbors for each point in the dataset.
    
    Attributes:
    - k: int, the number of nearest neighbors to find.
    """
    
    def __init__(self, k):
        pass
    
    def __call__(self, X):
        """        
        Parameters:
        - X: numpy array, the dataset (m x n).
        
        Returns:
        - neighbors: numpy array, adjacency matrix (m x m).
        """
        # TODO: For each point, find the indices of the k smallest distances
        return None
    

class EpsNeighborhood:
    """
    Compute the epsilon-neighborhood for each point in the dataset.
    
    Attributes:
    - epsilon: float, the maximum distance to consider a point as a neighbor.
    """
    
    def __init__(self, eps):
        pass
    
    def __call__(self, X):
        """
        Parameters:
        - X: numpy array, the dataset (m x n).
        
        Returns:
        - neighbors: numpy array, adjacency matrix (m x m).
        """
        # TODO: For each point, find the indices of points within the epsilon distance
        return None
