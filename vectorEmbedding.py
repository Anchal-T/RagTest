from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

from urlCrawl import extractFromUrl
from searchResult import getSearchUrls
load_dotenv()

def buildVecDb(query, urlNums=5):
    urls = getSearchUrls(query, urlNums)
    raw_content = extractFromUrl(urls)
    embaddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_texts(raw_content, embaddings, persist_directory='./chromaDb')
    print('Vector Db built')

if __name__ == "__main__":
    query = "How to bake a cake"
    buildVecDb(query, 5)