import chromadb
from sklearn.decomposition import PCA
import plotly.express as px

client = chromadb.PersistentClient(path="./chromaDb")
collection = client.get_collection("langchain")

# Fetch embeddings and the actual documents (text)
results = collection.get(include=['embeddings', 'documents'])
embeddings = results['embeddings']
docs = results['documents']

# Reduce the embedding dimensionality
pca = PCA(n_components=3)
vis_dims = pca.fit_transform(embeddings)

# Use short snippets for labels
short_docs = [' '.join(doc.split()[:8]) + "..." for doc in docs]

# Create an interactive 3D plot with text labels
fig = px.scatter_3d(
    x=vis_dims[:, 0],
    y=vis_dims[:, 1],
    z=vis_dims[:, 2],
    text=short_docs,  # Much faster!
    labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'z': 'PCA Component 3'},
    title='3D PCA of Embeddings'
)

fig.show()