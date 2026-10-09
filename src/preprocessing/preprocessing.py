
import numpy as np
from sklearn.preprocessing import StandardScaler
def log1p(data):
    data["Recency"] = np.log1p(data["Recency"])
    data["Frequency"] = np.log1p(data["Frequency"])
    data["Monetary"] = np.log1p(data["Monetary"])


def standarisation(data):
    columns = ["Recency","Frequency","Monetary"]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[columns])
    return data_scaled