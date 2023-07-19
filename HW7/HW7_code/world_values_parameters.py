import numpy as np

regression_knn_parameters = {
    'knn__n_neighbors': np.arange(1, 50),

    # Apply uniform weighting vs k for k Nearest Neighbors Regression
    ##### TODO(f): Change the weighting #####
    'knn__weights': ['uniform']
    # Uncomment the following code for part d)
    # 'knn__weights': ['distance']
}