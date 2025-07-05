from tavily import TavilyClient
from dotenv import load_dotenv
import os

from searchResult import getSearchUrls

load_dotenv()

query = "What are the best ways to refactor in python"
urls = getSearchUrls(query, 1)

client = TavilyClient(os.getenv("TAVILY_API_KEY"))
response = client.extract(
    urls=urls
)

raw_content = [item.get('raw_content', '') for item in response.get("results", [])]

print(raw_content)