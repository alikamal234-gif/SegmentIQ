import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.config.config import MODEL_PATH, insert_prediction
from src.cleaning.cleaning import nettoyage
from src.preprocessing.preprocessing import log1p

st.set_page_config(page_title="SegmentIQ", page_icon="📊", layout="wide")

st.title("SegmentIQ")
st.write("Segmentation des clients avec RFM + Random Forest")

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
noms_clusters = model_data["noms_clusters"]

st.sidebar.title("Mode de prédiction")
mode = st.sidebar.radio("Choisissez une méthode :", ["Importer un fichier Excel", "Saisie manuelle"])

if mode == "Importer un fichier Excel":
    st.header("Importer les données")
    uploaded_file = st.file_uploader("Choisissez un fichier Excel", type=["xlsx"])

    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.success("Fichier chargé avec succès.")
        st.write("### Données originales")
        st.dataframe(data.head(10), use_container_width=True)
        st.write(f"Nombre de lignes : **{len(data):,}**")

        rfm = nettoyage(data.copy())
        rfm_original = rfm.copy()

        rfm_model = rfm[["Recency", "Frequency", "Monetary"]].copy()
        log1p(rfm_model)

        predictions = model.predict(rfm_model)

        rfm_original["Nom_Cluster"] = predictions

        st.success(f"{len(rfm_original):,} clients ont été segmentés.")
        st.write("### Segmentation finale")
        st.dataframe(rfm_original, use_container_width=True)

        st.write("### Répartition des clients")
        cluster_counts = rfm_original["Nom_Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Nombre de clients"]
        st.dataframe(cluster_counts, use_container_width=True)

        csv = rfm_original.to_csv(index=False).encode("utf-8")
        st.download_button(label="Télécharger les résultats CSV", data=csv, file_name="segmentation_clients.csv", mime="text/csv")
else:
    st.header("Prédiction pour un seul client")
    st.write("Saisissez les valeurs RFM du client.")

    with st.form("prediction_form"):
        recency = st.number_input("Recency", min_value=0.0, value=30.0, step=1.0)
        frequency = st.number_input("Frequency", min_value=1.0, value=5.0, step=1.0)
        monetary = st.number_input("Monetary", min_value=0.0, value=1000.0, step=10.0)
        submitted = st.form_submit_button("Prédire le cluster")

    if submitted:
        new_data = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})
        
        new_data_model = new_data.copy()
        log1p(new_data_model)

        nom_cluster = model.predict(new_data_model)[0]

        data_final = {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "prediction": nom_cluster
        }
        insert_prediction(data_final)

        st.write("### Données du client")
        st.dataframe(new_data, use_container_width=True)

        st.write("### Résultat")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cluster", nom_cluster)
        st.success(f"Ce client appartient au segment : **{nom_cluster}**")