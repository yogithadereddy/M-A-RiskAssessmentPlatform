import sys
import os
import re
import nltk
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from database.mongodb_config import news_collection

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize Lemmatizer & Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_news_text(text):
    """Cleans financial news text by removing special characters, numbers, and stopwords."""
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove numbers & special characters
    tokens = word_tokenize(text)
    processed_text = " ".join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])
    
    return processed_text



def preprocess_and_store(company_name):
    """Fetch raw data from MongoDB, clean it, and store cleaned versions."""
    
    # Process news
    for news in news_collection.find({"company": company_name}):
        cleaned_text = clean_news_text(news["title"] + " " + news.get("description", ""))
        news_collection.update_one(
            {"_id": news["_id"]},
            {"$set": {"cleaned_text": cleaned_text}}
        )


    print(f" Preprocessed and stored cleaned data for {company_name}.")

# Test preprocessing
if __name__ == "__main__":
    company = input("Enter company name: ").strip()
    if company:
        preprocess_and_store(company)
    else:
        print(" Company name cannot be empty!")
