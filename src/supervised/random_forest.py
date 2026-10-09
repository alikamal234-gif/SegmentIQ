from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Sépare les données en train et test.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train,
    y_train,
    n_estimators=200,
    random_state=42
):
    """
    Entraîne le modèle Random Forest.
    """

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    rf_model.fit(
        X_train,
        y_train
    )

    return rf_model


def save_model(
    rf_model,
    noms_clusters,
    path="../models/rf_cluster_model.joblib"
):
    """
    Sauvegarde le modèle Random Forest
    et le mapping des clusters.
    """

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    model_data = {
        "model": rf_model,
        "noms_clusters": noms_clusters
    }

    joblib.dump(
        model_data,
        path
    )


def predict_cluster(rf_model, X):
    """
    Prédit le cluster de nouveaux clients.
    """

    return rf_model.predict(X)