from pca import PCA
from dataset import generate_plane

# Generate a noisy linear hyperplane dataset (adjust parameters as needed)
X, labels = generate_plane(n_samples=1000, n_classes=10, n_dim=2, noise=0.1, random_state=42)

pca = PCA(n_components=1)
X_transformed = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_transformed.shape)

X_reconstructed = pca.inverse_transform(X_transformed)
print("Reconstructed shape:", X_reconstructed.shape)

# Optionally visualize if needed
from dataset import visualize_plane
visualize_plane(X_transformed, labels)
pass