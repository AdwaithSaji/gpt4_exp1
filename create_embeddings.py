import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# ==============================
# Q3 - GENERATE FAQ EMBEDDINGS
# ==============================

# Load FAQ dataset
faq = pd.read_csv("data/faq.csv")

print("\n========== FAQ EMBEDDING GENERATION ==========")

# Load pre-trained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for FAQ questions
faq_embeddings = model.encode(
    faq["Question"].tolist(),
    convert_to_numpy=True
)

print("Number of FAQ questions:", len(faq))
print("Embedding shape:", faq_embeddings.shape)

# Create models folder if it doesn't exist
import os
os.makedirs("models", exist_ok=True)

# Store embeddings
with open("models/faq_embeddings.pkl", "wb") as file:
    pickle.dump(faq_embeddings, file)

print("\nFAQ embeddings generated successfully.")
print("Saved to: models/faq_embeddings.pkl")