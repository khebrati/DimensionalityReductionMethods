import numpy as np

# from pca import PCA
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

    X,labels = generate_plane(n_classes=6,noise = 0,n_dim=2)
    print(X)
    print(X.shape)
    visualize_plane(X,labels)
    
    pass
