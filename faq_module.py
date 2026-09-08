import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama


# ==============================
# FAQ MODULE
# Q5 + WEB APPLICATION VERSION
# ==============================

# Load FAQ dataset
faq = pd.read_csv("data/faq.csv")

# Load saved FAQ embeddings
with open("models/faq_embeddings.pkl", "rb") as file:
    faq_embeddings = pickle.load(file)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Similarity threshold
SIMILARITY_THRESHOLD = 0.50


# ==============================
# SEARCH FAQ
# ==============================

def search_faq(user_question):

    # Convert user question into embedding
    query_embedding = model.encode(
        [user_question],
        convert_to_numpy=True
    )

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_embedding,
        faq_embeddings
    )[0]

    # Find best matching FAQ
    best_index = np.argmax(similarity_scores)

    best_question = faq.iloc[best_index]["Question"]
    best_answer = faq.iloc[best_index]["Answer"]
    best_score = similarity_scores[best_index]

    return best_question, best_answer, float(best_score)


# ==============================
# GENERATE FAQ RESPONSE
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
        model="qwen2.5",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ==============================
# FAQ CHAT FUNCTION
# ==============================

def handle_faq(user_question):

    best_question, best_answer, best_score = search_faq(
        user_question
    )

    # Check similarity threshold
    if best_score >= SIMILARITY_THRESHOLD:

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

    else:

        return {
            "matched": False,
            "question": best_question,
            "answer": best_answer,
            "similarity": round(best_score, 2),
            "response": None
        }