import os
import pickle
import re

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

import ollama

from create_embeddings import build_product_vectors


# ==============================
# PRODUCT QA MODULE  (RAG)
# Q8 + Q9 + WEB APPLICATION
# ==============================

OLLAMA_MODEL = "qwen2.5"

# Load products dataset
products = pd.read_csv("data/products.csv")
products["Description"] = products["Description"].fillna("")


# ==============================
# LOAD TF-IDF MODEL
# ==============================

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
VECTORS_PATH = "models/product_vectors.pkl"

if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(VECTORS_PATH)):
    build_product_vectors()

with open(VECTORIZER_PATH, "rb") as file:
    tfidf = pickle.load(file)

with open(VECTORS_PATH, "rb") as file:
    product_vectors = pickle.load(file)


# Vocabulary used to decide whether a question names a product
BRANDS = {str(b).lower() for b in products["Brand"].unique()}
CATEGORIES = {str(c).lower() for c in products["Category"].unique()}


# ==============================
# HELPERS
# ==============================

def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def as_context(product):
    """Structured context passed to the LLM (Q8)."""

    return {
        "ProductID": int(product["ProductID"]),
        "ProductName": product["ProductName"],
        "Category": product["Category"],
        "Brand": product["Brand"],
        "Price": float(product["Price"]),
        "Description": product["Description"],
        "Rating": float(product["Rating"])
    }


# ==============================
# PRODUCT NAME MATCHING
# ==============================

def find_product_by_name(user_question):
    """
    Return the index of a product explicitly named in the question,
    or None. The longest matching name wins so that
    "Lenovo Laptop 10" is not matched as "Lenovo Laptop 1".
    """

    question = str(user_question).lower()
    question_tokens = tokenize(question)

    best_index = None
    best_length = 0

    for index, product in products.iterrows():

        product_name = str(product["ProductName"]).lower()

        # Exact phrase appears in the question, bounded by word
        # edges so that "Lenovo Laptop 1" does not match a
        # question about "Lenovo Laptop 10".
        phrase_match = re.search(
            r"\b" + re.escape(product_name) + r"\b",
            question
        ) is not None

        # Every word of the name appears as a separate token
        name_tokens = tokenize(product_name)
        token_match = name_tokens.issubset(question_tokens)

        if phrase_match or token_match:

            if len(product_name) > best_length:
                best_index = index
                best_length = len(product_name)

    return best_index


# ==============================
# FOLLOW-UP DETECTION
# ==============================

PRONOUNS = {"it", "its", "it's", "this", "that", "they", "them", "same"}

ATTRIBUTES = {
    "price", "cost", "costs", "brand", "rating", "rated",
    "category", "description", "expensive", "cheap", "much"
}


def is_follow_up(user_question):
    """
    True when the question refers back to the product that was
    already being discussed instead of naming a new one.

    A question such as "What is the price of Dell Laptop 5?" must
    NOT count as a follow-up just because it contains "price" -
    it names its own product.
    """

    tokens = tokenize(user_question)

    # The question mentions a brand or category, so it is
    # introducing a product of its own.
    if tokens & BRANDS or tokens & CATEGORIES:
        return False

    if tokens & PRONOUNS:
        return True

    # Bare attribute question: "What is the price?"
    if tokens & ATTRIBUTES:
        return True

    return False


# ==============================
# Q8 - PRODUCT RETRIEVAL
# ==============================

def retrieve_product(user_question):

    # First try exact product-name matching
    exact_index = find_product_by_name(user_question)

    if exact_index is not None:
        return as_context(products.iloc[exact_index]), 1.0

    # Otherwise fall back to TF-IDF retrieval
    query_vector = tfidf.transform([user_question])

    similarity_scores = cosine_similarity(
        query_vector,
        product_vectors
    )[0]

    ratings = products["Rating"].tolist()

    # Break exact-similarity ties with the customer rating
    best_index = min(
        range(len(similarity_scores)),
        key=lambda index: (
            -round(float(similarity_scores[index]), 6),
            -float(ratings[index])
        )
    )

    similarity = float(similarity_scores[best_index])

    if similarity <= 0:
        return None, 0.0

    return as_context(products.iloc[best_index]), similarity


# ==============================
# Q9 - OLLAMA PRODUCT QA
# ==============================

def answer_product_question(user_question, product_context):

    prompt = f"""
You are an e-commerce product assistant.

Answer the user's question using ONLY the product information
provided below.

PRODUCT CONTEXT:

Product ID: {product_context["ProductID"]}
Product Name: {product_context["ProductName"]}
Category: {product_context["Category"]}
Brand: {product_context["Brand"]}
Price: {product_context["Price"]} Indian Rupees
Description: {product_context["Description"]}
Rating: {product_context["Rating"]}

USER QUESTION:
{user_question}

IMPORTANT RULES:

- Use only the information in PRODUCT CONTEXT.
- Do not invent specifications or features.
- If the requested information is not available, say:
  "The requested information is not available in the product details."
- Give a short and clear answer.
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

def handle_product_question(user_question, previous_product=None):

    named_index = find_product_by_name(user_question)

    if named_index is not None:

        # The question names a product explicitly
        product = as_context(products.iloc[named_index])
        similarity = 1.0

    elif previous_product is not None and is_follow_up(user_question):

        # "What is its price?" - keep talking about the same product
        product = previous_product
        similarity = 1.0

    else:

        product, similarity = retrieve_product(user_question)

    if product is None:

        return {
            "product": None,
            "similarity": 0.0,
            "response": (
                "I could not find that product in the catalogue. "
                "Please mention the product name, for example "
                "\"Lenovo Laptop 3\"."
            )
        }

    response = answer_product_question(
        user_question,
        product
    )

    return {
        "product": product,
        "similarity": round(similarity, 2),
        "response": response
    }
