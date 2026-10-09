import pandas as pd


def profile_clusters(data_rfm):
    profiling = (
        data_rfm
        .groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
    )

    return profiling


def count_clients_per_cluster(data_rfm):

    counts = (
        data_rfm["Cluster"]
        .value_counts()
        .sort_index()
    )

    return counts


def create_original_rfm(data):
    """
    Recalcule le RFM avec les valeurs originales
    avant les transformations log.
    """

    transactions_clean = data.copy()

    # Nettoyage
    transactions_clean = transactions_clean.dropna(
        subset=["CustomerID"]
    )

    transactions_clean = transactions_clean[
        ~transactions_clean["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    ]

    transactions_clean = transactions_clean[
        (transactions_clean["Quantity"] > 0)
        & (transactions_clean["UnitPrice"] > 0)
    ]

    # Conversion de la date
    transactions_clean["InvoiceDate"] = pd.to_datetime(
        transactions_clean["InvoiceDate"]
    )

    # Calcul du revenu
    transactions_clean["Revenue"] = (
        transactions_clean["Quantity"]
        * transactions_clean["UnitPrice"]
    )

    # Date de référence
    date_reference = transactions_clean["InvoiceDate"].max()

    # RFM original
    rfm_original = (
        transactions_clean
        .groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x: (date_reference - x.max()).days
            ),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum")
        )
        .reset_index()
    )

    return rfm_original


def merge_rfm_with_clusters(rfm_original, data_rfm):
    """
    Associe le cluster de chaque client
    à ses valeurs RFM originales.
    """

    cluster_labels = data_rfm[
        ["CustomerID", "Cluster"]
    ].copy()

    rfm_profile = rfm_original.merge(
        cluster_labels,
        on="CustomerID",
        how="inner",
        validate="one_to_one"
    )

    return rfm_profile


def profile_original_rfm(rfm_profile):
    """
    Produit le profil détaillé de chaque cluster
    avec les valeurs RFM originales.
    """

    profiling_original = (
        rfm_profile
        .groupby("Cluster")
        .agg(
            Nombre_clients=("CustomerID", "nunique"),
            Recence_moyenne=("Recency", "mean"),
            Frequence_moyenne=("Frequency", "mean"),
            Montant_moyen=("Monetary", "mean"),
            Chiffre_affaires=("Monetary", "sum")
        )
        .reset_index()
    )

    return profiling_original


def calculate_revenue_contribution(profiling):
    """
    Calcule la contribution de chaque cluster
    au chiffre d'affaires total.
    """

    profiling = profiling.copy()

    total_revenue = profiling["Chiffre_affaires"].sum()

    profiling["Contribution_CA_%"] = (
        profiling["Chiffre_affaires"]
        / total_revenue
        * 100
    )

    profiling = profiling.round(2)

    return profiling