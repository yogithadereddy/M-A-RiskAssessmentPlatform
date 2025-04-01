M&A Risk Assessment Platform

Overview
The M&A Risk Assessment Platform is an AI-driven solution that evaluates the risk associated with Mergers and Acquisitions (M&A) by analyzing financial reports, legal documents, and market sentiment. It integrates financial forecasting, sentiment analysis, and legal risk assessment to provide a comprehensive risk score for potential M&A deals.

Tech Stack
Backend (Flask)
* Flask - API development
* TensorFlow/Keras - Financial risk prediction (LSTM model)
* VADER & BERT - Sentiment analysis on news & tweets
* Spacy & NER - Legal risk extraction from contracts
* MongoDB - NoSQL database for storing financial, sentiment, and legal risk data
Frontend (Streamlit)
* Streamlit - Interactive UI for risk visualization

Data Sources
* Yahoo Finance API - Financial reports & stock data
* NewsAPI - Financial news sentiment analysis
* Kaggle - Twitter Dataset for companies 

Project Flow
1. Financial Risk Analysis
1. Fetch Financial Data from Yahoo Finance API 
2. Preprocess Data - Normalize & handle missing values.
3. Train LSTM Model - Predict financial risk based on historical trends.
4. Risk Score - Calculate risk score
2. Sentiment Analysis
1. Collect Financial News using NewsAPI.
2. Analyze News Sentiment using BERT.
3. Collect Social Media Sentiment from Kaggle Twitter dataset 
4. Analyze Tweets using VADER to determine market perception.
5. Combine Sentiments (news + tweets) to form an overall sentiment score.
3. Legal Risk Assessment
1. Scrape Legal Documents from company reports and regulatory filings.
2. Apply NER (Named Entity Recognition) to extract legal risks.
3. Categorize Risks (Compliance Issues, Lawsuits, Regulatory Violations).
4. M&A Risk Dashboard
1. Financial, Sentiment & Legal Risk Scores are aggregated.
2. Risk Categories Assigned based on a weighted model.
3. Visualized in Streamlit Dashboard for interactive risk assessment.

Features
Financial Risk Analysis using LSTM model Sentiment Analysis from news & social media Legal Risk Extraction using NLP Comprehensive M&A Risk Score Interactive Dashboard for risk visualization

Future Enhancements
Report
Improved Dashboard
