import os
import pickle

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

import ollama

from create_embeddings import build_product_vectors


# ==============================
# PRODUCT RECOMMENDATION MODULE
# Q6 + Q7 + WEB APPLICATION
# ==============================

OLLAMA_MODEL = "qwen2.5"

# Load products dataset
products = pd.read_csv("data/products.csv")

# Handle missing descriptions
products["Description"] = products["Description"].fillna("")


# ==============================
# Q6 - LOAD TF-IDF MODEL
# ==============================

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
VECTORS_PATH = "models/product_vectors.pkl"

if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(VECTORS_PATH)):

    # Build them once instead of failing, so the chatbot works
    # on a fresh clone of the repository.
    build_product_vectors()

with open(VECTORIZER_PATH, "rb") as file:
    tfidf = pickle.load(file)

with open(VECTORS_PATH, "rb") as file:
    product_vectors = pickle.load(file)


# ==============================
# Q7 - RETRIEVE TOP PRODUCTS
# ==============================

def recommend_products(user_query, top_n=3):

    # Convert user query into a TF-IDF vector
    query_vector = tfidf.transform([user_query])

    # Calculate cosine similarity against every product
    similarity_scores = cosine_similarity(
        query_vector,
        product_vectors
    )[0]

    ratings = products["Rating"].tolist()

    # Every description in this dataset uses the same template,
    # so many products inside a category score exactly the same.
    # Customer rating is used to break those ties, otherwise the
    # "top 3" would just be whichever rows happen to come last.
    ranking = sorted(
        range(len(similarity_scores)),
        key=lambda index: (
            -round(float(similarity_scores[index]), 6),
            -float(ratings[index])
        )
    )

    results = []

    for index in ranking:

        # Skip products that share no term with the query
        if similarity_scores[index] <= 0:
            continue

        product = products.iloc[index]

        results.append({
            "ProductID": int(product["ProductID"]),
            "ProductName": product["ProductName"],
            "Category": product["Category"],
            "Brand": product["Brand"],
            "Price": float(product["Price"]),
            "Description": product["Description"],
            "Rating": float(product["Rating"]),
            "Similarity": float(similarity_scores[index])
        })

        if len(results) == top_n:
            break

    return results


# ==============================
# Q7 - OLLAMA RECOMMENDATION
# ==============================

def generate_recommendation(user_query, products_list):

    product_context = ""

    for product in products_list:

        product_context += f"""
Product Name: {product["ProductName"]}
Category: {product["Category"]}
Brand: {product["Brand"]}
Price: {product["Price"]}
Rating: {product["Rating"]}
Description: {product["Description"]}
Similarity Score: {product["Similarity"]:.2f}

"""

    prompt = f"""
You are an e-commerce product recommendation assistant.

The user asked:
{user_query}

Below are the top retrieved products:

{product_context}

Choose the BEST product from the retrieved products.

Give:
1. Product name
2. Brand
3. Price
4. A short reason why it is the best match

Use ONLY the information provided above.
Do not invent specifications or features.
Prices are in Indian Rupees.
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

def handle_recommendation(user_query):

    # Retrieve top 3 products
    retrieved_products = recommend_products(
        user_query,
        top_n=3
    )

    if not retrieved_products:

        return {
            "products": [],
            "response": (
                "I could not find any matching product. "
                "Try mentioning a category such as laptop, mobile, "
                "headphones or smartwatch."
            )
        }

    # Generate recommendation using Ollama
    response = generate_recommendation(
        user_query,
        retrieved_products
    )

    return {
        "products": retrieved_products,
        "response": response
    }
