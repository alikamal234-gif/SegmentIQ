import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "rf_cluster_model.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "data.xlsx")

def get_db_collection(db_name="SegmentIQ", collection_name="predictions"):
    client = MongoClient(MONGODB_URL)
    db = client[db_name]
    return db[collection_name]

def insert_prediction(data):
    collection = get_db_collection()
    return collection.insert_one(data)
