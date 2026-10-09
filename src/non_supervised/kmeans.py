from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def elbow_method(data_scaled, k_values=range(2, 11)):
    inertias = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        inertias.append(kmeans.inertia_)
    return inertias

def silhouette_method(data_scaled, k_values=range(2, 11)):
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        silhouette_scores.append(score)
    return silhouette_scores

def train_kmeans(data_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    return kmeans, labels

def assign_clusters(data, clusters):
    data = data.copy()
    data["Cluster"] = clusters
    return data

def kmeans_fc(data_scaled, data, n_clusters=3):
    k_values = range(2, 11)
    inertias = elbow_method(data_scaled, k_values)
    silhouette_scores = silhouette_method(data_scaled, k_values)
    kmeans, labels = train_kmeans(data_scaled, n_clusters)
    data_rfm = assign_clusters(data, labels)
    return kmeans, labels, data_rfm, inertias, silhouette_scores