import os
import sys
import requests
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import pipeline

# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from database.mongodb_config import news_collection  #  Import MongoDB collection
from backend.sentiment_analysis_model.fetch_news import fetch_news
from backend.sentiment_analysis_model.preprocessing import clean_news_text  # Import text preprocessing function

#  Load environment variables
load_dotenv()

#  Initialize BERT Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment(text):
    """Analyze sentiment using a BERT-based model and categorize it into sections."""
    result = sentiment_pipeline(text[:512])  #  BERT has a 512-token limit
    label = result[0]["label"]  
    score = result[0]["score"]

    # Convert BERT label to custom sentiment categories
    if "5" in label or "4" in label:
        sentiment_label = "Positive"
    elif "1" in label or "2" in label:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # Assign a sentiment section based on score
    if score >= 0.4:
        sentiment_section = "Positive News"
    elif 0.3 <= score < 0.4:
        sentiment_section = "Mixed News"
    else:
        sentiment_section = "Negative News"

    return sentiment_label, sentiment_section, score

def analyze_company_news_sentiment(company_name):
    """Fetch news for a company, preprocess text, and analyze sentiment using BERT."""
    print(f"🔍 Analyzing news sentiment for: {company_name} ...")
    
    #  Fetch news articles
    news_articles = fetch_news(company_name)

    if not news_articles:
        print(f"⚠️ No news articles found for {company_name}.")
        return pd.DataFrame()  

    #  Process news articles
    news_data = []
    sentiment_scores = []
    
    for article in news_articles:
        # Preprocess text before sentiment analysis
        raw_text = article["title"] + " " + article.get("description", "")
        cleaned_text = clean_news_text(raw_text)  # ✅ Preprocessing step

        sentiment_label, sentiment_section, sentiment_score = analyze_sentiment(cleaned_text)
        sentiment_scores.append(sentiment_score)

        news_data.append({
            "source": article["source"]["name"],
            "title": article["title"],
            "description": article.get("description", ""),
            "cleaned_text": cleaned_text,  # ✅ Store cleaned text
            "sentiment_label": sentiment_label,  # ✅ Positive, Neutral, Negative
            "sentiment_section": sentiment_section,  # ✅ Positive News, Neutral News, Negative News
            "sentiment_score": sentiment_score,
            "url": article["url"],
            "published_at": article["publishedAt"]
        })

    # ✅ Convert to DataFrame
    df_news = pd.DataFrame(news_data)

    # ✅ Compute average sentiment score
    if sentiment_scores:
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
        print(f"\n📊 **Average Sentiment Score for {company_name}: {avg_sentiment_score:.4f}**")
    else:
        print("\n⚠️ No valid sentiment scores computed.")

    return df_news

if __name__ == "__main__":
    company = input("Enter company name: ").strip()
    if company:
        news_df = analyze_company_news_sentiment(company)

        if not news_df.empty:
            # ✅ Calculate average sentiment score
            avg_sentiment_score = news_df["sentiment_score"].mean()

            # ✅ Determine overall sentiment label
            if avg_sentiment_score >= 0.6:
                overall_sentiment = "Positive"
            elif avg_sentiment_score < 0.4:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"

            # ✅ Save results to CSV in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(script_dir, f"{company}_news_sentiment.csv")
            news_df.to_csv(filename, index=False)

            # ✅ Print results
            print(f"\n📊 **Overall Sentiment for {company}: {overall_sentiment} ({avg_sentiment_score:.4f})**")
            print(f"📁 News sentiment analysis saved as '{filename}'.")
            print("\n🔹 News Sentiment Analysis:")
            print(news_df)
        else:
            print("⚠️ No news data available for analysis.")
    else:
        print("❌ Company name cannot be empty!")