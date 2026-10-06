import joblib
import pandas as pd
import os 
model_data = joblib.load(r"c:/Users/Youcode/Desktop/SegmentIQ/models/rf_cluster_model.joblib")
model = model_data["model"]

new_data = pd.DataFrame({
    "Recency": [3.5],
    "Frequency": [1.1],
    "Monetary": [6.5]
})

prediction = model.predict(new_data)[0]



print("Cluster :", prediction)