import numpy as np

from pca import PCA
# from isomap import Isomap
# from lle import LLE

# from geo import KNearestNeighbors, EpsNeighborhood
# from dataset import load_dataset
# from metrics import trustworthiness

from dataset import visualize_plane,generate_plane

if __name__ == "__main__":
    # TODO: load the swiss roll dataset
    
    # TODO: try with different dimensionality reduction
    # algorithms and different parameters
    
    # TODO: calculate trustworthiness for each combination
    
    # TODO: visualize and show the results

    X,labels = generate_plane(n_classes=3,noise = 0.2,n_dim=3)
    print(X)
    print(X.shape)
    visualize_plane(X,labels)
    pca = PCA(n_components=2)
    x_transformed = pca.fit_transform(X)
    recunstructed = pca.inverse_transform(x_transformed)
    visualize_plane(recunstructed,labels)
    
    pass
