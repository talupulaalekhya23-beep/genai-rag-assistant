import json
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chunking function
def chunk_text(text, chunk_size=350, overlap=50):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)

        if len(chunk_words) > 5:   # avoid very small chunks
            chunks.append(chunk)

    return chunks


# Load documents
with open("docs.json", "r") as f:
    docs = json.load(f)

vector_store = []

# Process each document
for doc in docs:

    title = doc["title"]
    content = doc["content"]

    # Split document into chunks
    chunks = chunk_text(content)

    for chunk in chunks:

        embedding = model.encode(chunk).tolist()

        vector_store.append({
            "title": title,
            "content": chunk,
            "embedding": embedding
        })


# Save embeddings
with open("vector_store.json", "w") as f:
    json.dump(vector_store, f)

print(f"Index built successfully with {len(vector_store)} chunks.")