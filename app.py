import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SegmentIQ",
    page_icon="📊",
    layout="wide"
)

st.title("SegmentIQ")
st.write("Segmentation des clients avec RFM + Random Forest")


# ============================================================
# 2. CHARGEMENT DU MODÈLE
# ============================================================

model_data = joblib.load(
    "C:/Users/Youcode/Desktop/SegmentIQ/models/rf_cluster_model.joblib"
)

model = model_data["model"]
noms_clusters = model_data["noms_clusters"]


# ============================================================
# 3. CHOIX DU MODE
# ============================================================

st.sidebar.title("Mode de prédiction")

mode = st.sidebar.radio(
    "Choisissez une méthode :",
    [
        "Importer un fichier Excel",
        "Saisie manuelle"
    ]
)


# ============================================================
# 4. MODE 1 : IMPORTER UN FICHIER EXCEL
# ============================================================

if mode == "Importer un fichier Excel":

    st.header("Importer les données")

    uploaded_file = st.file_uploader(
        "Choisissez un fichier Excel",
        type=["xlsx"]
    )

    if uploaded_file is not None:

        # ----------------------------------------------------
        # Lecture
        # ----------------------------------------------------

        data = pd.read_excel(uploaded_file)

        st.success("Fichier chargé avec succès.")

        st.write("### Données originales")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        st.write(
            f"Nombre de lignes : **{len(data):,}**"
        )


        # ----------------------------------------------------
        # Nettoyage
        # ----------------------------------------------------

        data_clean = data.copy()

        # CustomerID manquant
        data_clean = data_clean.dropna(
            subset=["CustomerID"]
        )

        # Factures annulées
        data_clean = data_clean[
            ~data_clean["InvoiceNo"]
            .astype(str)
            .str.startswith("C")
        ]

        # Quantity <= 0
        data_clean = data_clean[
            data_clean["Quantity"] > 0
        ]

        # UnitPrice <= 0
        data_clean = data_clean[
            data_clean["UnitPrice"] > 0
        ]


        # ----------------------------------------------------
        # Revenue
        # ----------------------------------------------------

        data_clean["Revenue"] = (
            data_clean["Quantity"]
            * data_clean["UnitPrice"]
        )


        # ----------------------------------------------------
        # InvoiceDate
        # ----------------------------------------------------

        data_clean["InvoiceDate"] = pd.to_datetime(
            data_clean["InvoiceDate"]
        )


        # ----------------------------------------------------
        # RFM
        # ----------------------------------------------------

        reference_date = data_clean["InvoiceDate"].max()

        rfm = (
            data_clean
            .groupby("CustomerID")
            .agg(
                Recency=(
                    "InvoiceDate",
                    lambda x: (
                        reference_date - x.max()
                    ).days
                ),

                Frequency=(
                    "InvoiceNo",
                    "nunique"
                ),

                Monetary=(
                    "Revenue",
                    "sum"
                )
            )
            .reset_index()
        )


        # ----------------------------------------------------
        # Sauvegarde des valeurs RFM originales
        # ----------------------------------------------------

        rfm_original = rfm.copy()


        # ----------------------------------------------------
        # Transformation log1p
        # ----------------------------------------------------

        rfm_model = rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ].copy()

        rfm_model[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ] = np.log1p(rfm_model)


        # ----------------------------------------------------
        # Prédiction
        # ----------------------------------------------------

        predictions = model.predict(
            rfm_model
        )


        # ----------------------------------------------------
        # Ajout des résultats
        # ----------------------------------------------------

        rfm_original["Cluster"] = predictions

        rfm_original["Nom_Cluster"] = (
            rfm_original["Cluster"]
            .map(noms_clusters)
        )


        # ----------------------------------------------------
        # Résultat
        # ----------------------------------------------------

        st.success(
            f"{len(rfm_original):,} clients ont été segmentés."
        )

        st.write("### Segmentation finale")

        st.dataframe(
            rfm_original,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Statistiques
        # ----------------------------------------------------

        st.write("### Répartition des clients")

        cluster_counts = (
            rfm_original["Nom_Cluster"]
            .value_counts()
            .reset_index()
        )

        cluster_counts.columns = [
            "Cluster",
            "Nombre de clients"
        ]

        st.dataframe(
            cluster_counts,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Télécharger le résultat
        # ----------------------------------------------------

        csv = rfm_original.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Télécharger les résultats CSV",
            data=csv,
            file_name="segmentation_clients.csv",
            mime="text/csv"
        )


# ============================================================
# 5. MODE 2 : SAISIE MANUELLE
# ============================================================

else:

    st.header("Prédiction pour un seul client")

    st.write(
        "Saisissez les valeurs RFM du client."
    )


    # --------------------------------------------------------
    # Formulaire
    # --------------------------------------------------------

    with st.form("prediction_form"):

        recency = st.number_input(
            "Recency",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

        frequency = st.number_input(
            "Frequency",
            min_value=1.0,
            value=5.0,
            step=1.0
        )

        monetary = st.number_input(
            "Monetary",
            min_value=0.0,
            value=1000.0,
            step=10.0
        )

        submitted = st.form_submit_button(
            "Prédire le cluster"
        )


    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------

    if submitted:

        # Données saisies par l'utilisateur
        new_data = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })


        # ----------------------------------------------------
        # Transformation log1p
        # ----------------------------------------------------

        new_data_model = new_data.copy()

        new_data_model[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ] = np.log1p(
            new_data_model[
                [
                    "Recency",
                    "Frequency",
                    "Monetary"
                ]
            ]
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            new_data_model
        )[0]


        # ----------------------------------------------------
        # Nom du cluster
        # ----------------------------------------------------

        nom_cluster = prediction



        # ----------------------------------------------------
        # Affichage
        # ----------------------------------------------------

        st.write("### Données du client")

        st.dataframe(
            new_data,
            use_container_width=True
        )


        st.write("### Résultat")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Cluster",
                prediction
            )

     

        st.success(
            f"Ce client appartient au segment : **{nom_cluster}**"
        )