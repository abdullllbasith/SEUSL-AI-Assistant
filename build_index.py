import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Step 1: Load your merged SEUSL JSON file
with open("seusl_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Extract content into text chunks
chunks = []

def extract_text(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            extract_text(value)
    elif isinstance(obj, list):
        for item in obj:
            extract_text(item)
    elif isinstance(obj, str):
        chunks.append(obj.strip())

extract_text(data)

# Step 3: Generate embeddings using sentence-transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)

# Step 4: Build and save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "seusl_index.faiss")

# Save the text chunks
with open("seusl_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ FAISS index and chunks saved successfully.")
