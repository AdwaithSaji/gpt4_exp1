import os
import pickle

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import ollama

from create_embeddings import (
    EMBEDDING_MODEL_NAME,
    build_faq_embeddings
)


# ==============================
# FAQ SEMANTIC SEARCH MODULE
# Q4 + Q5 + WEB APPLICATION
# ==============================

OLLAMA_MODEL = "qwen2.5"

EMBEDDINGS_PATH = "models/faq_embeddings.pkl"

# Below this cosine similarity the FAQ is treated as "no match"
# and the chatbot moves on to the next module (Q5).
SIMILARITY_THRESHOLD = 0.50


# Load FAQ dataset
faq = pd.read_csv("data/faq.csv")

# Load saved FAQ embeddings, building them on first run
if not os.path.exists(EMBEDDINGS_PATH):
    build_faq_embeddings()

with open(EMBEDDINGS_PATH, "rb") as file:
    faq_embeddings = pickle.load(file)

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ==============================
# Q4 - SEARCH THE FAQ
# ==============================

def search_faq(user_question):

    # Convert the user question into an embedding
    query_embedding = model.encode(
        [user_question],
        convert_to_numpy=True
    )

    # Cosine similarity against every stored FAQ question
    similarity_scores = cosine_similarity(
        query_embedding,
        faq_embeddings
    )[0]

    # Best matching FAQ
    best_index = int(np.argmax(similarity_scores))

    best_question = faq.iloc[best_index]["Question"]
    best_answer = faq.iloc[best_index]["Answer"]
    best_score = float(similarity_scores[best_index])

    return best_question, best_answer, best_score


# ==============================
# Q5 - GENERATE FAQ RESPONSE
# ==============================

def generate_response(user_question, faq_question, faq_answer):

    prompt = f"""
You are an e-commerce customer support assistant.

Answer the user's question using ONLY the information provided
in the FAQ answer.

FAQ Question:
{faq_question}

FAQ Answer:
{faq_answer}

User Question:
{user_question}

Give a short, natural and helpful response.
Do not add information that is not present in the FAQ answer.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ==============================
# CHATBOT / WEB APPLICATION
# ==============================

def handle_faq(user_question):

    best_question, best_answer, best_score = search_faq(
        user_question
    )

    # Q5 - apply the similarity threshold
    if best_score < SIMILARITY_THRESHOLD:

        return {
            "matched": False,
            "question": best_question,
            "answer": best_answer,
            "similarity": round(best_score, 2),
            "response": None
        }

    response = generate_response(
        user_question,
        best_question,
        best_answer
    )

    return {
        "matched": True,
        "question": best_question,
        "answer": best_answer,
        "similarity": round(best_score, 2),
        "response": response
    }
