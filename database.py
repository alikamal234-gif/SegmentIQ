from pymongo import MongoClient
import os
from dotenv import load_dotenv

def insert(data):
    load_dotenv()
    mongodb_url = os.getenv("MONGODB_URL")
    client = MongoClient(mongodb_url)
    db = client["SegmentIQ"]
    collection = db["predictions"]

    return collection.insert_one(data)