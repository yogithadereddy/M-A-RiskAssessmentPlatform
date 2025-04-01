from pymongo import MongoClient
import os
import datetime

# Connect to MongoDB on port 27018
client = MongoClient('mongodb://localhost:27018/')
db = client['risk_database']
collection = db['risk_data']
legal_risk_collection=db['legal_risk_analysis']
news_collection=db['news']
sentiment_collection=db['sentiment_analysis']

def store_risk_data(company_name, risk_score, model_accuracy):
    """Store risk data for a given company."""
    risk_entry = {
        "company_name": company_name,
        "risk_score": risk_score,
        "model_accuracy": model_accuracy,
        "timestamp": datetime.datetime.now()
    }
    collection.insert_one(risk_entry)
    print(f"✅ Risk data stored for {company_name}: {risk_score}")
