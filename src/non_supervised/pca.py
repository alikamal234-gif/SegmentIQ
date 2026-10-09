from sklearn.decomposition import PCA

def apply_pca(data_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(data_scaled)
    return pca, X_pca

def explained_variance(pca):
    return pca.explained_variance_ratio_

def cumulative_explained_variance(pca):
    return pca.explained_variance_ratio_.cumsum()