import numpy as np

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
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def _center_data(self, X):
        """
        Compute the mean of X along axis 0 (features) and subtract it from X.

        Parameters:
        - X: numpy array, shape (m, n)

        Returns:
        - centered X: numpy array, shape (m, n)
        """
        self.mean_ = np.mean(X, axis=0)
        return X - self.mean_

    def _create_cov(self, X):
        """
        Compute the covariance matrix of the centered dataset X.

        Covariance matrix formula:
            covariance_matrix = (X^T * X) / (m - 1)
        where m is the number of samples.

        Parameters:
        - X: numpy array, shape (m, n)

        Returns:
        - covariance matrix: numpy array, shape (n, n)
        """
        m = X.shape[0]
        return np.dot(X.T, X) / (m - 1)

    def _decompose(self, covariance_matrix):
        """
        Perform eigendecomposition using np.linalg.eigh.

        Parameters:
        - covariance_matrix: numpy array, shape (n, n)

        Returns:
        - eigvals: eigenvalues sorted in descending order
        - eigvecs: corresponding eigenvectors as columns, sorted accordingly
        """
        eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        return eigvals, eigvecs

    def fit(self, X):
        """
        Fit the PCA model to the dataset by computing the principal components.

        Parameters:
        - X: numpy array, the dataset (m x n).
        """
        # Center the data
        X_centered = self._center_data(X)
        # Compute the covariance matrix
        cov_matrix = self._create_cov(X_centered)
        # Perform eigendecomposition
        eigvals, eigvecs = self._decompose(cov_matrix)
        # Retain only the top n_components eigenvectors and eigenvalues
        self.components_ = eigvecs[:, :self.n_components]
        self.explained_variance_ = eigvals[:self.n_components]

    def transform(self, X):
        """
        Project the data onto the top principal components.

        Parameters:
        - X: numpy array, the data to project (m x n).

        Returns:
        - transformed data: numpy array, shape (m x n_components)
        """
        # Center X using the previously computed mean
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        """
        Fit the model to the data and return the projected data.

        Parameters:
        - X: numpy array, the data to transform (m x n).

        Returns:
        - transformed data: numpy array, shape (m x n_components)
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct the data from its PCA projection.

        Parameters:
        - X_transformed: numpy array, the PCA projection (m x n_components).

        Returns:
        - reconstructed data: numpy array, shape (m x n)
        """
        return np.dot(X_transformed, self.components_.T) + self.mean_


if __name__ == "__main__":
    # Example usage (pseudocode since dataset loading is project-specific):
    # from dataset import load_dataset
    # X, labels = load_dataset('swiss_roll.npz')
    #
    # pca = PCA(n_components=2)
    # X_transformed = pca.fit_transform(X)
    #
    # print("Original shape:", X.shape)
    # print("Transformed shape:", X_transformed.shape)
    #
    # X_reconstructed = pca.inverse_transform(X_transformed)
    # print("Reconstructed shape:", X_reconstructed.shape)
    pass


if __name__ == "__main__":
    # TODO: Load swiss roll dataset
    # TODO: Perform PCA
    # TODO: Visualize the results
    # TODO: Reconstruct dataset
    pass
