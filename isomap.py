import numpy as np
from dataset import generate_plane
from geo import KNearestNeighbors, _compute_distance_matrix  # Import our distance computation
from pca import PCA

class Isomap:
    """
    Isomap for dimensionality reduction by preserving geodesic distances.
    """

    def __init__(self, n_components, *, adj_calculator=KNearestNeighbors(5), decomposer=None):
        """
        Initialize Isomap with the number of components and neighbors.

        Parameters:
        - n_components: int, the number of dimensions to retain in the reduced space.
        - adj_calculator: callable, given a dataset returns the adjacency matrix.
        """
        self._adj_calculator = adj_calculator
        self._decomposer = decomposer or PCA(n_components=n_components)

    def _compute_geodesic_distances(self, X):
        """
        Compute the geodesic distance matrix using a shortest-path algorithm (Dijkstra's).

        For the given dataset X:
         1. Compute the full Euclidean distance matrix (using _compute_distance_matrix from geo.py).
         2. Use the adj_calculator (e.g., KNearestNeighbors) to obtain the binary adjacency matrix.
         3. Build a weighted graph where an edge exists if points are neighbors, with weight equal to the Euclidean distance.
         4. For each point, use Dijkstra's algorithm (with an outer loop) to compute the shortest (geodesic)
            distances to all other points.

        Parameters:
        - X: numpy array, shape (m x n)

        Returns:
        - geodesics: numpy array, shape (m x m) where geodesics[i, j] is the geodesic distance between points i and j.
        """
        m = X.shape[0]
        # Use the distance function from geo.py to compute the Euclidean distance matrix.
        D = _compute_distance_matrix(X)

        # Obtain the connectivity mask (neighbors) from the adj_calculator.
        adj = self._adj_calculator(X)
        # Build weighted graph: if connected use the Euclidean distance; otherwise, set to infinity.
        G = np.where(adj == 1, D, np.inf)
        # Ensure the diagonal is zero.
        np.fill_diagonal(G, 0)

        # Initialize geodesic distance matrix.
        geodesics = np.full((m, m), np.inf)
        
        # Compute geodesic distances using Dijkstra's algorithm for each starting point.
        for i in range(m):
            dist = G[i].copy()
            visited = np.zeros(m, dtype=bool)
            dist[i] = 0
            # Iterate over vertices
            for _ in range(m):
                # Select the unvisited vertex with the smallest tentative distance.
                j = np.argmin(np.where(visited, np.inf, dist))
                if np.isinf(dist[j]):
                    break
                visited[j] = True
                # Relax distances using vertex j.
                for k in range(m):
                    if not visited[k] and dist[j] + G[j, k] < dist[k]:
                        dist[k] = dist[j] + G[j, k]
            geodesics[i] = dist
        return geodesics

    def _decompose(self, geodesic_distances):
        """
        Convert geodesic distances into an inner product (Gram) matrix and apply PCA.

        The transformation is done by computing:
            D2 = element-wise squared geodesic distances (n x n)
            Centering matrix C = I - (1/n) J   (where J is an (n x n) matrix of ones)
            B = -1/2 * C * D2 * C
        B is the inner product matrix. PCA is then applied to B for dimensionality reduction.

        Parameters:
        - geodesic_distances: numpy array, (m x m) matrix.

        Returns:
        - Transformed data: numpy array, (m x n_components)
        """
        m = geodesic_distances.shape[0]
        D2 = geodesic_distances ** 2
        I = np.eye(m)
        J = np.ones((m, m))
        C = I - (1 / m) * J
        B = -0.5 * np.dot(np.dot(C, D2), C)
        # Apply PCA on the Gram matrix B.
        return self._decomposer.fit_transform(B)

    def fit_transform(self, X):
        """
        Fit the Isomap model to the dataset and reduce its dimensionality.

        Parameters:
        - X: numpy array, the dataset (m x n).

        Returns:
        - Low-dimensional embedding: numpy array, shape (m x n_components)
        """
        # Compute the geodesic distances.
        geodesic_distances = self._compute_geodesic_distances(X)
        # Decompose the inner product matrix (derived from the geodesic distances) to get embeddings.
        Y = self._decompose(geodesic_distances)
        return Y


if __name__ == "__main__":
    # Example usage:
    from dataset import load_dataset, visualize_plane

    # Load dataset: Here we use generate_plane to simulate a Swiss Roll-like structure.
    X, labels = generate_plane(n_classes=3, noise=0.1, n_dim=3, n_samples=100)
    
    # Visualize the original dataset.
    print("Visualizing original dataset...")
    visualize_plane(X, labels)
    
    # Apply Isomap for dimensionality reduction.
    isomap = Isomap(n_components=2, adj_calculator=KNearestNeighbors(10))
    X_transformed = isomap.fit_transform(X)
    
    # Visualize the 2D embedding.
    print("Visualizing Isomap-transformed data (2D)...")
    visualize_plane(X_transformed, labels)
    
    # (Optional) Additional analysis can be done here.
    pass