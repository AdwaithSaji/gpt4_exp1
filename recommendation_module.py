import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import ollama


# ==============================
# PRODUCT RECOMMENDATION MODULE
# Q6 + Q7 + WEB APPLICATION
# ==============================

# Load products dataset
products = pd.read_csv("data/products.csv")

# Handle missing descriptions
products["Description"] = products["Description"].fillna("")


# ==============================
# LOAD TF-IDF MODEL
# ==============================

if (
    os.path.exists("models/tfidf_vectorizer.pkl")
    and os.path.exists("models/product_vectors.pkl")
):

    with open("models/tfidf_vectorizer.pkl", "rb") as file:
        tfidf = pickle.load(file)

    with open("models/product_vectors.pkl", "rb") as file:
        product_vectors = pickle.load(file)

else:

    tfidf = TfidfVectorizer(stop_words="english")

    product_vectors = tfidf.fit_transform(
        products["Description"]
    )


# ==============================
# RETRIEVE TOP PRODUCTS
# ==============================

def recommend_products(user_query, top_n=3):

    # Convert user query into TF-IDF vector
    query_vector = tfidf.transform([user_query])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_vector,
        product_vectors
    )[0]

    # Get top products
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    results = []

    for index in top_indices:

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

    return results


# ==============================
# OLLAMA RECOMMENDATION
# ==============================

def generate_recommendation(user_query, products_list):

    product_context = ""

    for product in products_list:

        product_context += f"""
Product Name: {product["ProductName"]}
Category: {product["Category"]}
Brand: {product["Brand"]}
Price: ₹{product["Price"]}
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

def handle_recommendation(user_query):

    # Retrieve top 3 products
    retrieved_products = recommend_products(
        user_query,
        top_n=3
    )

    # Generate recommendation using Ollama
    response = generate_recommendation(
        user_query,
        retrieved_products
    )

    return {
        "products": retrieved_products,
        "response": response
    }