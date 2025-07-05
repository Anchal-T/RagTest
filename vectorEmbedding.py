from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

from urlCrawl import extractFromUrl
from searchResult import getSearchUrls
load_dotenv()

urls = getSearchUrls("How to bake a cake", 5)
raw_content = extractFromUrl(urls)

embaddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorDb = Chroma.from_texts(raw_content, embaddings, persist_directory='./chromaDb')

query = "at what temprature to bake a cake"
results = vectorDb.similarity_search(query, k=1)
for doc in results:
    print(doc.page_content)