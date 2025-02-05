import numpy as np

def generate_plane(n_samples=1000, n_classes=4, n_dim=3, noise=0.1, random_state=None):
    if n_dim < 2:
        raise ValueError("Number of dimensions must be at least 2")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    t = np.random.uniform(-1, 1, size=(n_samples, 2))
    points = [t[:, 0], t[:, 1]]
    
    for d in range(2, n_dim):
        new_dim = 0.5 * points[0] - 0.5 * points[1] + 0.2 * d
        points.append(new_dim)
    
    X = np.column_stack(points)
    X += np.random.normal(0, noise, size=(n_samples, n_dim))
    
    grid_size = int(np.ceil(np.sqrt(n_classes)))
    labels = np.zeros(n_samples, dtype=int)
    
    x_bins = np.linspace(X[:, 0].min(), X[:, 0].max(), grid_size + 1)
    y_bins = np.linspace(X[:, 1].min(), X[:, 1].max(), grid_size + 1)
    
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (
                (X[:, 0] >= x_bins[i]) & (X[:, 0] < x_bins[i + 1]) &
                (X[:, 1] >= y_bins[j]) & (X[:, 1] < y_bins[j + 1])
            )
            labels[mask] = i * grid_size + j
            if i * grid_size + j >= n_classes:
                labels[mask] = n_classes - 1
    
    return X, labels

def visualize_plane(X, labels):
    import matplotlib.pyplot as plt
    unique_labels = np.unique(labels)
    dims = X.shape[1]

    if dims == 2:
        plt.figure(figsize=(10, 8))
        for label in unique_labels:
            mask = labels == label
            plt.scatter(X[mask, 0], X[mask, 1], label=f'Class {label}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for label in unique_labels:
            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], label=f'Class {label}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

def load_dataset(path):
    dataset = np.load(path)
    return dataset['data'], dataset['target']
