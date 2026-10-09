import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_elbow(k_values, inertias):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Méthode du coude (Elbow Method)")
    plt.xticks(list(k_values))
    plt.grid(True)
    plt.show()

def plot_silhouette(k_values, silhouette_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, marker="o")
    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score selon le nombre de clusters")
    plt.xticks(list(k_values))
    plt.grid(True)
    plt.show()

def plot_clusters_pca(pca_df):
    plt.figure(figsize=(8, 6))
    for cluster in sorted(pca_df["Cluster"].unique()):
        subset = pca_df[pca_df["Cluster"] == cluster]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"Cluster {cluster}",
            alpha=0.6
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Visualisation des clusters avec PCA")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_importance(feature_names, importances):
    plt.figure(figsize=(8, 4))
    plt.bar(feature_names, importances, color='skyblue')
    plt.title("Importance des variables dans le Random Forest")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs réelles")
    plt.show()

def plot_cluster_distribution(cluster_counts):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
    plt.title("Distribution des clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Nombre de clients")
    plt.show()
