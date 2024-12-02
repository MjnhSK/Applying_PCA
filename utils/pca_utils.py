from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def k_components(target, df_train, threshold = 0.9):

    X = df_train.drop([target], axis = 1)

    # Standardize the Likert-scale responses
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    pca.fit_transform(X_scaled)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    # print("Explained Variance by Components:\n", explained_variance)

    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    # print("Cumulative Explained Variance:\n", cumulative_variance)

    # Choose k components (e.g., cumulative variance >= 90%)
    k = np.argmax(cumulative_variance >= threshold) + 1
    print("k =",k)
    return k