import pandas as pd
import numpy as np

# Imports de l'architecture
from src.config.config import DATA_PATH, MODEL_PATH
from src.cleaning.cleaning import nettoyage
from src.preprocessing.preprocessing import log1p, standarisation
from src.non_supervised.kmeans import train_kmeans, assign_clusters
from src.supervised.random_forest import split_data, train_random_forest, save_model

def run_training():
    print("=== Démarrage du pipeline d'entraînement ===")

    print("1. Chargement et Nettoyage des données...")
    data = pd.read_excel(DATA_PATH)
    data_rfm = nettoyage(data)
    
    print("2. Preprocessing (Log1p et Standardisation)...")
    data_rfm_log = data_rfm.copy()
    log1p(data_rfm_log)
    data_scaled = standarisation(data_rfm_log)

    print("3. Entraînement K-Means...")
    kmeans, labels = train_kmeans(data_scaled, n_clusters=3)
    data_rfm_clustered = assign_clusters(data_rfm_log, labels)

    noms_clusters = {
        0: "Clients Inactifs",
        1: "Clients Premium",
        2: "Clients Réguliers"
    }
    data_rfm_clustered["Nom_Cluster"] = data_rfm_clustered["Cluster"].map(noms_clusters)

    print("4. Entraînement du Random Forest...")
    X = data_rfm_clustered[["Recency", "Frequency", "Monetary"]]
    y = data_rfm_clustered["Nom_Cluster"]

    X_train, X_test, y_train, y_test = split_data(X, y)
    rf_model = train_random_forest(X_train, y_train)

    print(f"5. Sauvegarde du modèle dans {MODEL_PATH}...")
    save_model(rf_model, noms_clusters, MODEL_PATH)

    print("=== Entraînement terminé avec succès ! ===")

if __name__ == "__main__":
    run_training()
