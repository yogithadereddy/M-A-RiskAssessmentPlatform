import spacy
from spacy.matcher import PhraseMatcher
from transformers import pipeline
from newspaper import Article  # To extract full text from URLs
import logging
from backend.legal_and_regulatory_risk_model.legal_news_filter import fetch_filtered_urls
import requests

# Suppress HuggingFace and TensorFlow warnings for cleaner console output
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load spaCy's pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Load pre-trained summarization pipeline from HuggingFace (BART model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize PhraseMatcher for detecting fine-related terms
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
fine_terms = ["fine", "fined", "penalty", "penalized", "violated", "legal settlement"]
patterns = [nlp.make_doc(term) for term in fine_terms]
matcher.add("FINE_TERMS", patterns)


def extract_named_entities(text):
    """
    Extracts named entities from the given text using spaCy.
    Focuses on entities like ORGANIZATION, MONEY, DATE, LAW, and other relevant labels.
    Adds custom FINE entities for terms like "fined" or "penalty."
    """
    doc = nlp(text)
    entities = []

    # Add standard NER entities (ORG, MONEY, etc.)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "LAW", "MONEY", "DATE", "GPE", "PERCENT", "EVENT", "NORP"]:
            entities.append((ent.text, ent.label_))

    # Apply custom matcher for fine-related terms
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        entities.append((span.text, "FINE"))  # Add "FINE" entity

    return entities


def summarize_text(text):
    """
    Summarizes the given text using HuggingFace's BART summarizer.
    The summary length can be tuned with min_length and max_length.
    """
    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Summarization failed due to text length or pipeline error."


def analyze_legal_article(text):
    """
    Analyzes the legal article by extracting named entities from both the full text and summary.
    Prints the key entities and the summary of the article.
    """
    # Step 1: Named Entity Recognition (NER) on the full article
    full_article_entities = extract_named_entities(text)

    # Step 2: Summarization and NER on summarized text
    summary = summarize_text(text)
    summary_entities = extract_named_entities(summary)

    # Output the results
    print("\nNamed Entities Extracted from Full Article:")
    for ent in full_article_entities:
        print(f"{ent[0]} -> {ent[1]}")

    print("\nNamed Entities Extracted from Summary:")
    for ent in summary_entities:
        print(f"{ent[0]} -> {ent[1]}")

    print("\nSummary of the Article:")
    print(summary)


def extract_full_article(url):
    """
    Extracts the text of a full article from the given URL using newspaper3k.
    Includes headers to bypass potential bot blocks.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise error for bad status codes

        # Use newspaper3k to parse the HTML content
        article = Article(url)
        article.set_html(response.text)
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error extracting article from {url}: {e}")
        return None


if __name__ == "__main__":
    # Ask the user for a company name
    company_name = input("Enter the company name to analyze legal news articles: ").strip()
    article_urls = fetch_filtered_urls(company_name)

    if article_urls:
        # Loop through URLs, extract, analyze, and summarize each article
        for url in article_urls:
            print(f"\nAnalyzing Article: {url}")
            article_text = extract_full_article(url)
            if article_text:
                analyze_legal_article(article_text)
            else:
                print("Failed to extract article text. Skipping...\n")
    else:
        print(f"No legal articles found for {company_name}.")
