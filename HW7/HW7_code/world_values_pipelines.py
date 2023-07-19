from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

k_nearest_neighbors_regression_pipeline = Pipeline(
        [
            # Apply scaling to k Nearest Neighbors Regression
            ##### TODO(g): Add a 'scale' parameter that applies StandardScaler() #####
            # Uncomment the following line for part g)
            # ('scale', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ]
    )