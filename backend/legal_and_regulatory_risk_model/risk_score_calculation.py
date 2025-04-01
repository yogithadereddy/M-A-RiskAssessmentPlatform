from backend.legal_and_regulatory_risk_model import ner  # Importing functions from ner.py
from collections import Counter
from backend.legal_and_regulatory_risk_model.legal_news_filter import fetch_filtered_urls  # To fetch URLs based on company name


def calculate_legal_risk_score(entities):
   
    risk_weights = {
        "ORG": 0.2,        
        "LAW": 0.4,         
        "MONEY": 0.4,       
        "DATE": 0.1,        
        "GPE": 0.2,         
        "EVENT": 0.3,       
        "CARDINAL": 0.1,    
        "FINE": 0.5,       
        "PENALTY": 0.5,     
        "CRIME": 0.6,       
        "CLAIM": 0.4        
    }

    entity_counts = Counter([label for _, label in entities])
    risk_score = 0.0

    for entity_label, count in entity_counts.items():
        weight = risk_weights.get(entity_label, 0)
        risk_score += weight * min(count, 5)

    normalized_risk_score = min(risk_score / 3.5, 1.0)
    return normalized_risk_score


def display_risk_analysis(url, text):
    print(f"\n--- Legal Risk Analysis for Article: {url} ---")

    # Analysis for full article
    full_entities = ner.extract_named_entities(text)
    full_risk_score = calculate_legal_risk_score(full_entities)
    
    # Display full article results
    print("\nNamed Entities Extracted from the Full Article:")
    full_entity_counts = Counter([label for _, label in full_entities])
    for ent, count in full_entity_counts.items():
        print(f"{ent}: {count} occurrences")
    print(f"\nLegal Risk Score (Full Article) (0 to 1): {full_risk_score:.2f}")

    # Summarized text analysis
    summary = ner.summarize_text(text)
    summary_entities = ner.extract_named_entities(summary)
    summary_risk_score = calculate_legal_risk_score(summary_entities)

    # Display summarized text results
    print("\nNamed Entities Extracted from the Summarized Text:")
    summary_entity_counts = Counter([label for _, label in summary_entities])
    for ent, count in summary_entity_counts.items():
        print(f"{ent}: {count} occurrences")
    print(f"\nLegal Risk Score (Summarized Text) (0 to 1): {summary_risk_score:.2f}")
    
    print(f"\nSummary of the Article:\n{summary}")


if __name__ == "__main__":
    company_name = input("Enter the company name to analyze legal news articles: ").strip()
    article_urls = fetch_filtered_urls(company_name)

    if article_urls:
        for url in article_urls:
            article_text = ner.extract_full_article(url)
            if article_text:
                display_risk_analysis(url, article_text)
            else:
                print(f"Failed to extract text from {url}. Skipping...\n")
    else:
        print(f"No legal articles found for {company_name}.")
