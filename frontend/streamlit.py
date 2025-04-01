import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading errors
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Backend API Base URL
API_BASE_URL = "http://127.0.0.1:5000"

# Streamlit Page Configuration
st.set_page_config(
    page_title="M&A AI Platform Dashboard",
    layout="wide",
)

def calculate_overall_risk(financial_risk, legal_risk, sentiment_score):
    
    # Assign practical weights
    financial_weight = 0.5
    legal_weight = 0.3
    sentiment_weight = 0.2
    
    # Calculate weighted risk
    overall_risk = (
        financial_weight * financial_risk + 
        legal_weight * legal_risk + 
        sentiment_weight * sentiment_score
    )
    
    return overall_risk



def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = st.sidebar.radio("Go to", ["Home", "Financial Risk", "Legal Risk", "Sentiment Analysis","Overall M&A Risk"])

    # --- Home Page ---
    if pages == "Home":
        st.title("M&A AI Platform Dashboard")
        st.write(
            "Welcome to the M&A AI Risk Assessment Platform! Use the navigation menu to explore financial risk analysis, "
            "legal risk review, and sentiment insights for different companies."
        )
        st.image(
            "/Users/kdn_aidyogitha/Desktop/projects/M&A-AI-Platform/frontend/Adobe Express - file.jpg",
            use_container_width=True
        )

        st.markdown("""
            ### Key Features:
            - **Financial Risk Analysis**: Analyze and predict financial risk scores using LSTM.
            - **Legal Risk Assessment**: Review and assess legal risks from articles related to M&A.
            - **Sentiment Analysis**: Get sentiment scores from news and tweets.
        """)

    # --- Financial Risk Dashboard ---
    elif pages == "Financial Risk":
        st.title("Financial Risk Analysis Dashboard")
        st.markdown("Explore **financial risk trends, company insights, and LSTM model analysis.**")

        st.subheader("Enter Company Name to Calculate Risk Score")
        company_name = st.text_input("Company Name")

        if st.button("Calculate Financial Risk"):
            if not company_name.strip():
                st.warning("Please enter a valid company name.")
            else:
                try:
                    response = requests.post(f"{API_BASE_URL}/calculate_risk", json={"company_name": company_name})

                    if response.status_code == 200:
                        risk_data = response.json()
                        st.success(f"Risk data for {company_name} successfully calculated!")
                        st.write(f"**Risk Score:** {risk_data['risk_score']:.4f}")
                        st.write(f"**Model Accuracy:** {risk_data['model_accuracy']:.2f}")

                        if "history" in risk_data:
                            performance_data = pd.DataFrame(risk_data["history"])
                            plt.figure(figsize=(10, 5))
                            plt.plot(performance_data["epoch"], performance_data["loss"], label="Loss")
                            plt.plot(performance_data["epoch"], performance_data["accuracy"], label="Accuracy")
                            plt.xlabel("Epochs")
                            plt.ylabel("Loss / Accuracy")
                            plt.title("Training Performance")
                            plt.legend()
                            st.pyplot(plt)

                    elif response.status_code == 400:
                        st.error("Company name is missing!")
                    else:
                        st.error(f"Error: {response.json().get('error', 'Something went wrong.')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API Request Error: {e}")

        st.subheader("View Financial Risk Data for All Companies")
        if st.button("Fetch Risk Data"):
            try:
                response = requests.get(f"{API_BASE_URL}/get_risk_data")
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        df = pd.DataFrame(data)
                        st.write("### Financial Risk Data", df)

                        st.write("### Descriptive Statistics")
                        st.write(df.describe())

                        st.write("### Visual Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            plt.figure(figsize=(6, 4))
                            sns.histplot(df["risk_score"], bins=10, kde=True, color="blue")
                            plt.xlabel("Risk Score")
                            plt.title("Distribution of Risk Scores")
                            st.pyplot(plt)

                        with col2:
                            top_5_companies = df.sort_values(by="risk_score", ascending=False).head(5)
                            st.write("Top 5 Companies with Highest Risk Scores")
                            st.write(top_5_companies)
                    else:
                        st.info("No financial risk data available.")
                else:
                    st.error("Failed to fetch financial risk data.")
            except requests.exceptions.RequestException as e:
                st.error(f"API Request Error: {e}")


    # --- Legal Risk Assessment ---
    elif pages == "Legal Risk":
        st.title("Legal Risk Assessment")
        st.markdown("""
        Analyze and visualize legal risk based on related articles. 
        This dashboard provides average risk scores, entity distribution, and article-level insights.
        """)

        company_name = st.text_input("Company Name for Legal Risk Assessment")

        if st.button("Analyze Legal Risk"):
            if not company_name.strip():
                st.warning("Please enter a valid company name.")
            else:
                try:
                    response = requests.post(f"{API_BASE_URL}/analyze-legal-risk", json={"company_name": company_name})
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Legal risk analysis completed!")
                        st.write(f"**Average Risk Score:** {result['average_risk_score']:.2f}")
                        st.write(f"**Articles Analyzed:** {result['articles_analyzed']}")

                        # Convert article results to DataFrame for analysis
                        articles_df = pd.DataFrame(result["results"])

                        # --- Data Insights Section ---
                        st.subheader("Article Insights")
                        st.write("### Top 3 Articles with Highest Full Risk Scores")
                        top_articles = articles_df.nlargest(3, "full_risk_score")
                        st.table(top_articles[["url", "full_risk_score", "summary_risk_score"]])


                        # Bar Chart: Full Risk vs Summary Risk
                        st.write("### Article Risk Scores")
                        fig, ax = plt.subplots(figsize=(10, 5))

                        # Create shorter labels like 'Article 0', 'Article 1', etc.
                        article_labels = [f"Article {i}" for i in range(len(articles_df))]

                        # Plot the bars using these shorter labels instead of full URLs
                        ax.bar(article_labels, articles_df["full_risk_score"], color='red', alpha=0.6, label='Full Risk Score')
                        ax.bar(article_labels, articles_df["summary_risk_score"], color='blue', alpha=0.6, label='Summary Risk Score')

                        ax.set_xlabel("Articles")
                        ax.set_ylabel("Risk Score")
                        ax.set_title("Comparison of Full and Summary Risk Scores")
                        plt.xticks(rotation=45, ha="right")
                        ax.legend()

                        st.pyplot(fig)


                        # --- Entity Distribution Pie Chart ---
                        st.write("### Entity Distribution from Article Summaries")
                        all_summary_entities = [label for sublist in articles_df["summary_entities"] for label, _ in sublist]
                        entity_counts = Counter(all_summary_entities)

                        if entity_counts:
                            fig, ax = plt.subplots()
                            ax.pie(entity_counts.values(), labels=entity_counts.keys(), autopct='%1.1f%%', startangle=140)
                            ax.set_title("Entity Type Distribution")
                            st.pyplot(fig)
                        else:
                            st.info("No entities found in the article summaries.")

                        # Interactive DataTable for detailed article info
                        st.write("### Detailed Article Data")
                        st.dataframe(articles_df[["url", "full_risk_score", "summary_risk_score", "summary_text"]])

                        # --- Descriptive Statistics ---
                        st.write("### Descriptive Statistics for Risk Scores")
                        st.write(articles_df[["full_risk_score", "summary_risk_score"]].describe())

                    elif response.status_code == 404:
                        st.error(f"No articles found for {company_name}.")
                    else:
                        st.error("Something went wrong. Try again later.")
                except requests.exceptions.RequestException as e:
                    st.error(f"API Request Error: {e}")





    # --- Sentiment Analysis ---
    elif pages == "Sentiment Analysis":
        st.title("Sentiment Analysis for M&A Decisions")
        company_name = st.text_input("Company Name")
        ticker = st.text_input("Stock Ticker")

        if st.button("Analyze Sentiment"):
            if not company_name.strip() or not ticker.strip():
                st.warning("Please enter both company name and stock ticker.")
            else:
                try:
                    response = requests.post(f"{API_BASE_URL}/analyze_sentiment", json={"company_name": company_name, "ticker": ticker})
                    if response.status_code == 201:
                        sentiment_data = response.json()["data"]
                        st.success(f"Sentiment analysis for {company_name} completed!")
                        st.write(f"**Combined Sentiment Score:** {sentiment_data['combined_sentiment_score']:.2f}")
                        st.write(f"**Sentiment Label:** {sentiment_data['sentiment_label']}")
                    elif response.status_code == 200:
                        sentiment_data = response.json()["data"]
                        st.warning("Sentiment data already exists for this company.")
                        st.write("Existing Sentiment Data:")
                        st.write(sentiment_data)
                    else:
                        st.error(f"Unexpected Error: {response.status_code}. Response: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API Request Error: {e}")



    elif pages == "Overall M&A Risk":
        st.title("Overall M&A Risk Assessment")
        st.markdown("""
        Calculate a combined risk score by analyzing **financial risk**, **legal risk**, and **sentiment**.
        Enter the company name and stock ticker for a comprehensive assessment.
        """)

        company_name = st.text_input("Company Name")
        ticker = st.text_input("Stock Ticker")

        if st.button("Calculate Overall Risk"):
            if not company_name.strip() or not ticker.strip():
                st.warning("Please enter both company name and stock ticker.")
            else:
                try:
                    # Get financial risk score
                    financial_response = requests.post(f"{API_BASE_URL}/calculate_risk", json={"company_name": company_name})
                    financial_risk = financial_response.json().get("risk_score", 0.0)

                    # Get legal risk score
                    legal_response = requests.post(f"{API_BASE_URL}/analyze-legal-risk", json={"company_name": company_name})
                    legal_risk = legal_response.json().get("average_risk_score", 0.0)

                    # Get sentiment score
                    sentiment_response = requests.post(f"{API_BASE_URL}/analyze_sentiment", json={"company_name": company_name, "ticker": ticker})
                    sentiment_data = sentiment_response.json().get("data", {})
                    sentiment_score = sentiment_data.get("combined_sentiment_score", 0.0)

                    # Calculate overall risk
                    overall_risk = calculate_overall_risk(financial_risk, legal_risk, sentiment_score)

                    # Display results
                    st.success(f"Overall M&A Risk Score for {company_name} Calculated!")
                    st.write(f"**Financial Risk Score:** {financial_risk:.2f}")
                    st.write(f"**Legal Risk Score:** {legal_risk:.2f}")
                    st.write(f"**Sentiment Score:** {sentiment_score:.2f}")
                    st.write(f"### **Overall Risk Score:** {overall_risk:.2f}")

                    # Display risk interpretation
                    if overall_risk >= 0.7:
                        st.error("High Risk: The M&A deal is potentially risky.")
                    elif 0.4 <= overall_risk < 0.7:
                        st.warning("Moderate Risk: The M&A deal may involve some risk.")
                    else:
                        st.success("Low Risk: The M&A deal seems relatively safe.")

                except requests.exceptions.RequestException as e:
                    st.error(f"API Request Error: {e}")


if __name__ == '__main__':
    main()
