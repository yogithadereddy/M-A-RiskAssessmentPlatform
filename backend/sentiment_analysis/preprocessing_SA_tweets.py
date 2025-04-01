import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

tweets = pd.read_csv('/Users/kdn_aidyogitha/Desktop/projects/M&A-AI-Platform/backend/sentiment_analysis_model/Combined_sampled_dataset.csv')

tweets['date'] = pd.to_datetime(tweets['post_date'], unit='s').dt.date

sia = SentimentIntensityAnalyzer()

positive_words = 'high profit Growth Potential Opportunity Bullish Strong Valuable Success Promising Profitable Win Winner Outstanding Record Earnings Breakthrough buy bull long support undervalued underpriced cheap upward rising trend moon rocket hold breakout call beat support buying holding'
negative_words = 'resistance squeeze cover seller Risk Loss Decline Bearish Weak Declining Uncertain Troubling Downturn Struggle Unstable Volatile Slump Disaster Plunge sell bear bubble bearish short overvalued overbought overpriced expensive downward falling sold sell low put miss'

dictOfpos = {word: 4 for word in positive_words.split(" ")}
dictOfneg = {word: -4 for word in negative_words.split(" ")}
Financial_Lexicon = {**dictOfpos, **dictOfneg}
sia.lexicon.update(Financial_Lexicon)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text

def get_sentiment(tweets, ticker):
    df = tweets.loc[tweets['ticker_symbol'] == ticker].copy()
    
    if 'post_date' in df.columns and 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['post_date'], unit='s').dt.date
    
    df['body'] = df['body'].apply(clean_text)
    df['score'] = df['body'].apply(lambda x: sia.polarity_scores(x)['compound'])
    avg_score = df['score'].mean()
    sentiment_label = 'Positive' if avg_score > 0.1 else 'Negative' if avg_score < -0.1 else 'Neutral'
    print(f"\nOverall Sentiment Score for {ticker}: {avg_score:.2f} ({sentiment_label})\n")
    df['label'] = pd.cut(df['score'], bins=[-1, -0.66, 0.32, 1], labels=["bad", "neutral", "good"])
    return df[['date', 'score', 'label', 'tweet_id', 'body']], avg_score, sentiment_label

def plot_sentiment_distribution(tw, company_name):
    sentiment_counts = tw.groupby(['date', 'label'], observed=False).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'good': 'green', 'neutral': 'grey', 'bad': 'red'}
    sentiment_counts.plot.area(stacked=True, ax=ax, color=[colors[c] for c in sentiment_counts.columns])
    handles = [mpatches.Patch(color=colors[label], label=label.capitalize()) for label in sentiment_counts.columns]
    ax.legend(handles=handles, title='Sentiment', loc="upper left")
    plt.title(f'Sentiment Distribution by Date for {company_name}')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.grid(True)
    plt.show()

def plot_daily_sentiment(tw, company_name):
    daily_sentiment = tw.groupby('date')['score'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sentiment.index, daily_sentiment.values, label="Average Sentiment Score", color='b')
    plt.axhline(y=0, color='r', linestyle='--', label="Neutral Sentiment")
    plt.title(f'Average Daily Sentiment Score for {company_name}')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("Available Tickers: ", tweets['ticker_symbol'].unique())
    ticker = input("Enter a ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
    company_name = tweets.loc[tweets['ticker_symbol'] == ticker, 'company_name'].iloc[0]
    tw, avg_score, sentiment_label = get_sentiment(tweets, ticker)
    print(f"\nShowing sentiment analysis for {company_name} ({ticker})\n")
    plot_sentiment_distribution(tw, company_name)
    plot_daily_sentiment(tw, company_name)

if __name__ == "__main__":
    main()
