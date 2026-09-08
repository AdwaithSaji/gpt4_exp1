import pandas as pd
import pickle
import re

from sklearn.metrics.pairwise import cosine_similarity
import ollama


# ==============================
# PRODUCT QA MODULE
# Q8 + Q9 + WEB APPLICATION
# ==============================

# Load products dataset
products = pd.read_csv("data/products.csv")
products["Description"] = products["Description"].fillna("")


# ==============================
# LOAD TF-IDF MODEL
# ==============================

with open("models/tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("models/product_vectors.pkl", "rb") as file:
    product_vectors = pickle.load(file)


# ==============================
# PRODUCT NAME MATCHING
# ==============================

def find_product_by_name(user_question):

    question = user_question.lower()

    # Check exact product names first
    for index, product in products.iterrows():

        product_name = str(product["ProductName"]).lower()

        if product_name in question:

            return index


    # Check brand + product number
    for index, product in products.iterrows():

        product_name = str(product["ProductName"]).lower()

        words = product_name.split()

        if all(word in question for word in words):

            return index

    return None


# ==============================
# PRODUCT RETRIEVAL
# ==============================

def retrieve_product(user_question):

    # First try exact product-name matching
    exact_index = find_product_by_name(user_question)

    if exact_index is not None:

        product = products.iloc[exact_index]

        context = {
            "ProductID": int(product["ProductID"]),
            "ProductName": product["ProductName"],
            "Category": product["Category"],
            "Brand": product["Brand"],
            "Price": float(product["Price"]),
            "Description": product["Description"],
            "Rating": float(product["Rating"])
        }

        return context, 1.0


    # Otherwise use TF-IDF retrieval
    query_vector = tfidf.transform([user_question])

    similarity_scores = cosine_similarity(
        query_vector,
        product_vectors
    )[0]

    best_index = similarity_scores.argmax()

    product = products.iloc[best_index]

    similarity = similarity_scores[best_index]

    context = {
        "ProductID": int(product["ProductID"]),
        "ProductName": product["ProductName"],
        "Category": product["Category"],
        "Brand": product["Brand"],
        "Price": float(product["Price"]),
        "Description": product["Description"],
        "Rating": float(product["Rating"])
    }

    return context, float(similarity)


# ==============================
# OLLAMA PRODUCT QA
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
Price: ₹{product_context["Price"]}
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
# WEB APPLICATION FUNCTION
# ==============================

def handle_product_question(user_question, previous_product=None):

    # Handle follow-up questions
    follow_up_words = [
        "its",
        "it",
        "this",
        "that",
        "price",
        "brand",
        "rating",
        "category"
    ]

    is_follow_up = any(
        word in user_question.lower().split()
        for word in follow_up_words
    )

    if is_follow_up and previous_product is not None:

        product = previous_product
        similarity = 1.0

    else:

        product, similarity = retrieve_product(
            user_question
        )

    response = answer_product_question(
        user_question,
        product
    )

    return {
        "product": product,
        "similarity": round(similarity, 2),
        "response": response
    }