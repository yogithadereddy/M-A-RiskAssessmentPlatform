from newsapi import NewsApiClient

# Initialize NewsAPI with your API key
newsapi = NewsApiClient(api_key='3a87b64f6e85453cb9491b87db1ee95c')

LEGAL_KEYWORDS = [
    "lawsuit", "litigation", "class-action", "compliance", "antitrust", 
    "regulatory fine", "SEC investigation", "fraud", "financial misconduct", "data privacy violation"
]

def fetch_filtered_urls(company_name, required_articles=5, max_pages=3, page_size=30):
    """
    Fetches legal and regulatory news articles about the specified company and returns their URLs.
    Ensures at least 'required_articles' articles are fetched while respecting API limits (max 100 results).
    """
    try:
        query = f"{company_name} AND ({' OR '.join(LEGAL_KEYWORDS)})"
        legal_urls = []
        page = 1

        while len(legal_urls) < required_articles and page <= max_pages:
            # Fetch articles within the allowed range
            articles = newsapi.get_everything(
                q=query, language='en', sort_by='relevancy', page_size=page_size, page=page
            )

            # Check API response and handle errors
            if articles['status'] != 'ok':
                print("Error fetching articles:", articles.get('message', 'Unknown error'))
                break

            if articles['totalResults'] == 0:
                print(f"No legal news articles found for {company_name}.")
                break

            # Filter articles with company name in title or description
            for article in articles['articles']:
                if company_name.lower() in article['title'].lower() or company_name.lower() in article['description'].lower():
                    legal_urls.append(article['url'])

            page += 1  # Go to the next page

        # Return only the required number of articles
        return legal_urls[:required_articles]

    except Exception as e:
        print(f"An error occurred while fetching articles: {e}")
        return []


if __name__ == "__main__":
    company_name = input("Enter the name of the company you want to search legal news for: ").strip()
    urls = fetch_filtered_urls(company_name)

    if urls:
        print(f"\nLegal news articles for {company_name}:")
        for url in urls:
            print(f"URL: {url}")
    else:
        print(f"\nNo relevant legal news articles found for {company_name}.")
