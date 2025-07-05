from httpcore import __name
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

def getSearchUrls(query, maxResult=5):
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        max_results=maxResult
    )
    return [item["url"] for item in response.get("results", [])]

if __name__ == "__main__":
    query = "What are the best ways to refactor in python"
    urls = getSearchUrls(query, 5)
