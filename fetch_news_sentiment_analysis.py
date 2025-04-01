import os
import sys
import requests
from dotenv import load_dotenv

# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from database.mongodb_config import news_collection  # ✅ Import from root-level database

# ✅ Load API Keys
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_news(company_name, num_articles=50):
    """Fetch latest financial news for the given company and store in MongoDB."""
    if not NEWS_API_KEY:
        print("❌ Missing News API Key! Check your .env file.")
        return []

    company_name = company_name.strip()
    if not company_name:
        print("❌ Company name cannot be empty!")
        return []

    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&apiKey={NEWS_API_KEY}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        if not articles:
            print(f"❌ No news articles found for {company_name}.")
            return []

        # ✅ Store in MongoDB
        for article in articles[:num_articles]:
            news_collection.insert_one({
                "company": company_name,
                "source": article["source"]["name"],
                "title": article["title"],
                "description": article.get("description", ""),
                "url": article["url"],
                "published_at": article["publishedAt"]
            })
        
        print(f"✅ Stored {len(articles[:num_articles])} news articles in MongoDB.")
        return articles[:num_articles]

    except requests.exceptions.RequestException as e:
        print(f"❌ API request error: {e}")
        return []
