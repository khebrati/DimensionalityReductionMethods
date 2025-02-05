import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    """

    def __init__(self, n_components):
        """
        Initialize PCA with the number of components to retain.

        Parameters:
        - n_components: int, the number of principal components to keep.
        """
        # TODO: initialize required instance variables.

    def _center_data(self, X):
        # TODO: Compute the mean of X along axis 0 (features) and subtract it from X
        return None

    def _create_cov(self, X):
        # TODO: Use the formula for the covariance matrix.
        return None

    def _decompose(self, covariance_matrix):
        # TODO: Use np.linalg.eigh to get eigenvalues and eigenvectors
        return None

    def fit(self, X):
        """
        Fit the PCA model to the dataset by computing the principal components.

        Parameters:
        - X: numpy array, the centered dataset (m x n).
        """
        # TODO: Center the data

        # TODO: Compute the covariance matrix

        # TODO: Perform eigendecomposition
        pass
    
    def transform(self, X):
        """
        Project the data onto the top principal components.

        Parameters:
        - X: numpy array, the data to project (m x n).

        Returns:
        - transformed_data: numpy array, the data projected onto the top principal components.
        """
        # TODO: Center the data

        # TODO: Apply projection
        return None

    def fit_transform(self, X):
        """
        Fit the PCA model and transform the data in one step.

        Parameters:
        - X: numpy array, the data to fit and transform (m x n).

        Returns:
        - transformed_data: numpy array, the data projected onto the top principal components.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct the original data from the transformed data.

        Parameters:
        - X_transformed: numpy array, the data in the reduced dimensional space.

        Returns:
        - original_data: numpy array, the reconstructed data in the original space.
        """
        # TODO: Apply reconstruction formula
        return None


if __name__ == "__main__":
    # TODO: Load swiss roll dataset
    # TODO: Perform PCA
    # TODO: Visualize the results
    # TODO: Reconstruct dataset
    pass
