import os
import pandas as pd
from backend.sentiment_analysis_model.sentiment_analysis import analyze_company_news_sentiment
from backend.sentiment_analysis_model.preprocessing_SA_tweets import get_sentiment, tweets

def calculate_combined_sentiment(news_score, tweet_score, weight_news=0.6, weight_tweets=0.4):
    tweet_score_normalized = (tweet_score + 1) / 2
    combined_score = (weight_news * news_score) + (weight_tweets * tweet_score_normalized)
    if combined_score > 0.6:
        overall_sentiment = "Positive"
    elif 0.4 <= combined_score <= 0.6:
        overall_sentiment = "Neutral"
    else:
        overall_sentiment = "Negative"
    return combined_score, overall_sentiment

news_sentiment_score = 0.6
tweet_sentiment_score = 0.3
combined_score, sentiment_label = calculate_combined_sentiment(news_sentiment_score, tweet_sentiment_score)
print(f"Combined Sentiment Score: {combined_score:.2f} ({sentiment_label})")

def main():
    company_name = input("Enter company name: ").strip()
    ticker = input("Enter a ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
    if company_name and ticker:
        news_df = analyze_company_news_sentiment(company_name)
        tw, tweet_avg_score, tweet_sentiment_label = get_sentiment(tweets, ticker)
        if not news_df.empty:
            news_avg_score = news_df["sentiment_score"].mean()
            print(f"\n\ud83d\udcca Average Sentiment Score for News: {news_avg_score:.4f}")
        else:
            print("\u26a0\ufe0f No news articles found, using a default neutral score.")
            news_avg_score = 0.5
        print(f"\n\ud83d\udcca Average Sentiment Score for Tweets: {tweet_avg_score:.4f} ({tweet_sentiment_label})")
        combined_score, combined_sentiment_label = calculate_combined_sentiment(news_avg_score, tweet_avg_score)
        print(f"\n\ud83d\udd2e Combined Sentiment Score: {combined_score:.4f} ({combined_sentiment_label})")
        output_filename = f"{company_name}_combined_sentiment.csv"
        combined_data = {
            "Source": ["News", "Tweets", "Combined"],
            "Average Sentiment Score": [news_avg_score, tweet_avg_score, combined_score],
            "Sentiment Label": [
                "Positive" if news_avg_score >= 0.6 else "Negative" if news_avg_score < 0.4 else "Neutral",
                tweet_sentiment_label,
                combined_sentiment_label,
            ],
        }
        combined_df = pd.DataFrame(combined_data)
        combined_df.to_csv(output_filename, index=False)
        print(f"\n\ud83d\udcc1 Combined sentiment analysis saved as '{output_filename}'.")
        print(combined_df)
    else:
        print("\u274c Company name or ticker cannot be empty!")

if __name__ == "__main__":
    main()
