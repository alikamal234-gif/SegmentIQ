
import pandas as pd
def remove_NaN(data):
    return data.dropna(subset=["CustomerID"])

def remove_canclled(data):
    return data[~data["InvoiceNo"].astype(str).str.startswith("C")]



def fix_quantity_prix(data):
    return data[
        (data["Quantity"] > 0) &
        (data["UnitPrice"] > 0)
    ]



def rfm(data):
    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    date_reference = data["InvoiceDate"].max()
    rfm = data.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (date_reference - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()
    return rfm


def nettoyage(data):
    new_data = remove_NaN(data)
    new_data = remove_canclled(new_data)
    new_data = fix_quantity_prix(new_data)
    new_data = rfm(new_data)
    return new_data