from tavily import TavilyClient
from dotenv import load_dotenv
import os

from searchResult import getSearchUrls

load_dotenv()


def extractFromUrl(urls):
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    response = client.extract(
        urls=urls
    )
    return [item.get('raw_content', '') for item in response.get("results", [])]


if __name__ == "__main__":
    query = "What are the best ways to refactor in python"
    urls = getSearchUrls(query, 1)
    
    raw_content = extractFromUrl(urls)
    print(raw_content)