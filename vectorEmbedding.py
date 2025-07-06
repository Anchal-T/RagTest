from transformers.masking_utils import chunked_overlay
from h11._abnf import chunk_size
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter
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

    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    allChunks = []
    for doc in raw_content:
        allChunks.extend(textSplitter.split_text(doc))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_texts(allChunks, embeddings, persist_directory='./chromaDb')
    print('Vector Db built')

if __name__ == "__main__":
    query = "How to bake a cake"
    buildVecDb(query, 5)