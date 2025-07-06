from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def query_vector_db(query, k=1):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorDb = Chroma(persist_directory='./chromaDb', embedding_function=embeddings)
    results = vectorDb.similarity_search(query, k=k)
    for doc in results:
        print(doc.page_content)

if __name__ == "__main__":
    query = "How to refactor"
    query_vector_db(query, k=1)