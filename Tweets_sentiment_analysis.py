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

# Download VADER lexicon if it's not already downloaded
nltk.download('vader_lexicon')

# Load the dataset (ensure the CSV path is correctly set)
tweets = pd.read_csv('/Users/kdn_aidyogitha/Desktop/projects/M&A-AI-Platform/backend/sentiment_analysis_model/Combined_sampled_dataset.csv')

# Convert post_date from Unix timestamp to human-readable date and time
tweets['date'] = pd.to_datetime(tweets['post_date'], unit='s').dt.date

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Define financial-specific positive and negative keywords
positive_words = 'high profit Growth Potential Opportunity Bullish Strong Valuable Success Promising Profitable Win Winner Outstanding Record Earnings Breakthrough buy bull long support undervalued underpriced cheap upward rising trend moon rocket hold breakout call beat support buying holding'
negative_words = 'resistance squeeze cover seller Risk Loss Decline Bearish Weak Declining Uncertain Troubling Downturn Struggle Unstable Volatile Slump Disaster Plunge sell bear bubble bearish short overvalued overbought overpriced expensive downward falling sold sell low put miss'

# Create lexicon dictionaries and update the VADER lexicon with financial-specific words
dictOfpos = {word: 4 for word in positive_words.split(" ")}
dictOfneg = {word: -4 for word in negative_words.split(" ")}
Financial_Lexicon = {**dictOfpos, **dictOfneg}
sia.lexicon.update(Financial_Lexicon)

def clean_text(text):
    """
    Clean the input text by removing URLs, mentions, and converting to lowercase.
    Args:
        text (str): The input tweet text.
    Returns:
        str: Cleaned text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    return text

def get_sentiment(tweets, ticker):
    """Perform sentiment analysis on tweets for a given ticker symbol and calculate average sentiment."""
    # Filter tweets for the given ticker and ensure 'date' is present
    df = tweets.loc[tweets['ticker_symbol'] == ticker].copy()
    
    if 'post_date' in df.columns and 'date' not in df.columns:
        # Convert Unix timestamp to human-readable date if not already done
        df['date'] = pd.to_datetime(df['post_date'], unit='s').dt.date
    
    # Apply text cleaning before sentiment scoring
    df['body'] = df['body'].apply(clean_text)
    
    # Calculate sentiment scores using VADER
    df['score'] = df['body'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Calculate average sentiment score and label
    avg_score = df['score'].mean()
    sentiment_label = 'Positive' if avg_score > 0.1 else 'Negative' if avg_score < -0.1 else 'Neutral'
    print(f"\nOverall Sentiment Score for {ticker}: {avg_score:.2f} ({sentiment_label})\n")
    
    # Create sentiment labels based on score
    df['label'] = pd.cut(df['score'], bins=[-1, -0.66, 0.32, 1], labels=["bad", "neutral", "good"])
    
    return df[['date', 'score', 'label', 'tweet_id', 'body']], avg_score, sentiment_label


def plot_sentiment_distribution(tw, company_name):
    """
    Plot sentiment distribution over time for the given company's tweets.
    Args:
        tw (DataFrame): DataFrame with sentiment labels and dates.
        company_name (str): Name of the company.
    """
    # Group by date and label to get daily sentiment counts
    sentiment_counts = tw.groupby(['date', 'label'], observed=False).size().unstack(fill_value=0)

    # Plot area chart for sentiment distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'good': 'green', 'neutral': 'grey', 'bad': 'red'}
    sentiment_counts.plot.area(stacked=True, ax=ax, color=[colors[c] for c in sentiment_counts.columns])

    # Add a legend for sentiment labels
    handles = [mpatches.Patch(color=colors[label], label=label.capitalize()) for label in sentiment_counts.columns]
    ax.legend(handles=handles, title='Sentiment', loc="upper left")

    # Customize chart labels and title
    plt.title(f'Sentiment Distribution by Date for {company_name}')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.grid(True)
    plt.show()

def plot_daily_sentiment(tw, company_name):
    """
    Plot the average daily sentiment score for the given company's tweets.
    Args:
        tw (DataFrame): DataFrame with sentiment scores and dates.
        company_name (str): Name of the company.
    """
    # Calculate average sentiment score per day
    daily_sentiment = tw.groupby('date')['score'].mean()

    # Plot line chart for daily sentiment scores
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
    # Display available tickers to the user
    print("Available Tickers: ", tweets['ticker_symbol'].unique())
    
    # Prompt the user to enter a ticker symbol
    ticker = input("Enter a ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()

    # Retrieve the company name for the given ticker
    company_name = tweets.loc[tweets['ticker_symbol'] == ticker, 'company_name'].iloc[0]

    # Perform sentiment analysis and generate sentiment plots
    tw, avg_score, sentiment_label = get_sentiment(tweets, ticker)
    print(f"\nShowing sentiment analysis for {company_name} ({ticker})\n")
    plot_sentiment_distribution(tw, company_name)
    plot_daily_sentiment(tw, company_name)

if __name__ == "__main__":
    main()
