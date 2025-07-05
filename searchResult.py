from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

client = TavilyClient(os.getenv("TAVILY_API_KEY"))
response = client.search(
    query="What are the best practices for implementing RAG in production?",
    max_results=10
)
