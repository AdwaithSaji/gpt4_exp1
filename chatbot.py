import pandas as pd

# ==============================
# Q1 - LOAD DATASETS
# ==============================

products = pd.read_csv("data/products.csv")
faq = pd.read_csv("data/faq.csv")
reviews = pd.read_csv("data/reviews.csv")


# ==============================
# DISPLAY FIRST 5 RECORDS
# ==============================

print("\n========== PRODUCTS DATASET ==========")
print(products.head())

print("\n========== FAQ DATASET ==========")
print(faq.head())

print("\n========== REVIEWS DATASET ==========")
print(reviews.head())


# ==============================
# DISPLAY ROWS AND COLUMNS
# ==============================

print("\n========== DATASET SHAPES ==========")

print("Products:", products.shape)
print("FAQ:", faq.shape)
print("Reviews:", reviews.shape)


# ==============================
# DISPLAY ATTRIBUTES
# ==============================

print("\n========== DATASET ATTRIBUTES ==========")

print("\nProducts Columns:")
print(products.columns.tolist())

print("\nFAQ Columns:")
print(faq.columns.tolist())

print("\nReviews Columns:")
print(reviews.columns.tolist())