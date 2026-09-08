import pandas as pd
from transformers import pipeline
import ollama


# ==============================
# REVIEW INTELLIGENCE MODULE
# Q10 + Q11 + WEB APPLICATION
# ==============================

# Load reviews dataset
reviews = pd.read_csv("data/reviews.csv")

# Load pre-trained sentiment model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


# ==============================
# ANALYZE PRODUCT REVIEWS
# ==============================

def analyze_reviews(product_name):

    product_reviews = reviews[
        reviews["ProductName"].str.lower() == product_name.lower()
    ]

    if product_reviews.empty:
        return None

    positive = 0
    neutral = 0
    negative = 0

    detailed_reviews = []

    for review in product_reviews["Review"]:

        result = sentiment_analyzer(review)[0]

        label = result["label"]
        score = result["score"]

        if label == "POSITIVE":
            sentiment = "Positive"
            positive += 1

        elif label == "NEGATIVE":
            sentiment = "Negative"
            negative += 1

        else:
            sentiment = "Neutral"
            neutral += 1

        detailed_reviews.append({
            "Review": review,
            "Sentiment": sentiment,
            "Confidence": float(score)
        })

    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "details": detailed_reviews
    }


# ==============================
# OLLAMA REVIEW SUMMARY
# ==============================

def generate_review_summary(product_name, review_data):

    review_text = ""

    for item in review_data["details"][:30]:

        review_text += f"""
Review: {item["Review"]}
Sentiment: {item["Sentiment"]}
Confidence: {item["Confidence"]:.3f}
"""

    prompt = f"""
You are an e-commerce review analysis assistant.

Product:
{product_name}

Sentiment Summary:
Positive: {review_data["positive"]}
Neutral: {review_data["neutral"]}
Negative: {review_data["negative"]}

Customer Reviews:
{review_text}

Generate a concise review summary.

Include exactly these sections:

Overall Opinion
Strengths
Weaknesses
Buying Suggestion

Use only the information provided above.
Do not invent product features or customer opinions.
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

def handle_review_question(product_name):

    review_data = analyze_reviews(product_name)

    if review_data is None:

        return {
            "found": False,
            "response": f"No reviews were found for {product_name}."
        }

    summary = generate_review_summary(
        product_name,
        review_data
    )

    return {
        "found": True,
        "positive": review_data["positive"],
        "neutral": review_data["neutral"],
        "negative": review_data["negative"],
        "details": review_data["details"],
        "response": summary
    }