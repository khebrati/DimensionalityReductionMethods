import numpy as np

from dataset import load_dataset, visualize_plane
from pca import PCA

def main():

    X, labels = load_dataset("datasets\swissroll.npz")
    print(X.shape)
    print(X)
    print(labels.shape)
    print(labels)
    
    # Visualize the original Swiss Roll dataset.
    # If X has 3 or more dimensions, visualize using 3D; otherwise, use 2D.
    print("Visualizing original Swiss Roll dataset...")
    # visualize_plane(X, labels)
    
    # Apply PCA to reduce the dimensionality to 2 components.
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    
    # Visualize the 2D projection from PCA.
    print("Visualizing PCA-transformed data (2D)...")
    visualize_plane(X_transformed, labels)
    
    # Reconstruct the data from the 2D representation.
    X_reconstructed = pca.inverse_transform(X_transformed)
    # print("Reconstructed data shape:", X_reconstructed.shape)
    
    # Compute reconstruction error (mean squared error).
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print("Reconstruction Mean Squared Error:", reconstruction_error)

if __name__ == "__main__":
    main()