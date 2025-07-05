from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

raw_content = [
    "Stripe Payments is a global payment processing platform.",
    "Another document about refactoring in Python."
]

embaddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorDb = Chroma.from_texts(raw_content, embaddings, persist_directory='./chromaDb')

query = "What is gay"
results = vectorDb.similarity_search(query, k=2)
for doc in results:
    print(doc.page_content)