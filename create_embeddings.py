import os
import pickle

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ==============================
# EMBEDDING / VECTOR BUILDER
# Q3 - FAQ EMBEDDINGS
# Q6 - PRODUCT TF-IDF VECTORS
# ==============================

DATA_DIR = "data"
MODELS_DIR = "models"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ==============================
# PRODUCT SEARCH TEXT
# ==============================

def build_product_corpus(products):
    """
    Build the text used to represent each product.

    The Description column alone is not enough for this dataset,
    because every description follows the same template:

        "Laptop by Lenovo suitable for everyday use ..."

    With only descriptions, a query such as "Samsung mobile"
    cannot separate one product from another.

    The product name, category and brand are therefore combined
    with the description so that TF-IDF has real terms to match.
    """

    return (
        products["ProductName"].astype(str)
        + " "
        + products["Category"].astype(str)
        + " "
        + products["Brand"].astype(str)
        + " "
        + products["Description"].astype(str)
    )


# ==============================
# Q3 - GENERATE FAQ EMBEDDINGS
# ==============================

def build_faq_embeddings():

    # Imported here so that the TF-IDF step can still run
    # on machines where sentence-transformers is not installed.
    from sentence_transformers import SentenceTransformer

    print("\n========== FAQ EMBEDDING GENERATION ==========")

    faq = pd.read_csv(os.path.join(DATA_DIR, "faq.csv"))

    # Load pre-trained Sentence Transformer model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Convert every FAQ question into a numerical vector
    faq_embeddings = model.encode(
        faq["Question"].tolist(),
        convert_to_numpy=True
    )

    print("Number of FAQ questions :", len(faq))
    print("Embedding shape         :", faq_embeddings.shape)

    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, "faq_embeddings.pkl"), "wb") as file:
        pickle.dump(faq_embeddings, file)

    print("\nFAQ embeddings generated successfully.")
    print("Saved to: models/faq_embeddings.pkl")

    return faq_embeddings


# ==============================
# Q6 - GENERATE PRODUCT TF-IDF
# ==============================

def build_product_vectors():

    print("\n========== PRODUCT TF-IDF GENERATION ==========")

    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    products["Description"] = products["Description"].fillna("")

    corpus = build_product_corpus(products)

    # token_pattern keeps single-character tokens so that the
    # trailing product number ("Lenovo Laptop 3") stays in the
    # vocabulary. The default pattern requires two or more
    # characters and silently drops those digits, which makes
    # every product in a category look identical.
    tfidf = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b\w+\b"
    )

    product_vectors = tfidf.fit_transform(corpus)

    print("Number of products      :", len(products))
    print("Vocabulary size         :", len(tfidf.vocabulary_))
    print("Vector matrix shape     :", product_vectors.shape)

    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as file:
        pickle.dump(tfidf, file)

    with open(os.path.join(MODELS_DIR, "product_vectors.pkl"), "wb") as file:
        pickle.dump(product_vectors, file)

    print("\nProduct TF-IDF vectors generated successfully.")
    print("Saved to: models/tfidf_vectorizer.pkl")
    print("Saved to: models/product_vectors.pkl")

    return tfidf, product_vectors


# ==============================
# RUN BOTH STEPS
# ==============================

def main():

    build_faq_embeddings()
    build_product_vectors()

    print("\nAll models are ready.")


if __name__ == "__main__":
    main()
