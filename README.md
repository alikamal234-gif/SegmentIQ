SegmentIQ
=========

Projet de segmentation client basé sur RFM (Recency, Frequency, Monetary) avec K-Means et Random Forest.

## Architecture

* `data/` : Contient les données brutes (ex: `data.xlsx`).
* `models/` : Modèles pré-entraînés (ex: `rf_cluster_model.joblib`).
* `notebooks/` :
  * `main.ipynb` : Pipeline complet d'orchestration.
  * `visualisation.ipynb` : Notebook d'exploration visuelle.
* `src/` :
  * `cleaning/` : Fonctions de nettoyage et RFM.
  * `preprocessing/` : Transformation (Log1p, Standardisation).
  * `non_supervised/` : Algorithmes non supervisés (K-Means, PCA, Profiling).
  * `supervised/` : Modèle de classification (Random Forest, CV, Evaluation).
  * `utils/` : Fonctions utilitaires, notamment toutes les visualisations avec Matplotlib/Seaborn.
  * `streamlit/` : Application de démonstration.
  * `config/` : Configuration (chemins, MongoDB, variables d'environnement).

## Utilisation

1. Installez les dépendances : `pip install -r requirements.txt`
2. Lancez l'interface Streamlit : `streamlit run src/streamlit/app.py`
