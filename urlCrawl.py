import json
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


def cache_raw_content(query, url_count=1, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{query.replace(' ', '_')}_{url_count}_raw.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    urls = getSearchUrls(query, url_count)
    raw_content = extractFromUrl(urls)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(raw_content, f)
    return raw_content


if __name__ == "__main__":
    query = "What are the best ways to refactor in python"
    raw_content = cache_raw_content(query, 1)