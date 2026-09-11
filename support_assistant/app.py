# Module 3 - Zepto Support Assistant

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


# Paths
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load all 8 documents
documents = []
ids = []

for i in range(1, 9):
    file_path = DOCS_DIR / f"doc_{i:02d}.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().strip()

    documents.append(text)
    ids.append(f"doc_{i:02d}")


print(f"Loaded {len(documents)} documents.")


# Create embeddings
embeddings = model.encode(documents).tolist()

print("Embeddings created successfully.")


# Create persistent ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="zepto_support_docs"
)


# Store documents and embeddings
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)


print("Documents stored in ChromaDB successfully!")
print("Collection count:", collection.count())