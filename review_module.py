import pandas as pd

from transformers import pipeline

import ollama


# ==============================
# REVIEW INTELLIGENCE MODULE
# Q10 + Q11 + WEB APPLICATION
# ==============================

OLLAMA_MODEL = "qwen2.5"

SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# The SST-2 model only emits POSITIVE / NEGATIVE. Reviews such as
# "Average quality" land near the decision boundary, so anything the
# model is not confident about is reported as Neutral. Lower this
# value to make the Neutral bucket smaller.
NEUTRAL_CONFIDENCE_THRESHOLD = 0.90

# Reviews included in the prompt sent to Ollama
MAX_REVIEWS_IN_PROMPT = 30


# Load reviews dataset
reviews = pd.read_csv("data/reviews.csv")

# Q10 - Load pre-trained sentiment analysis model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL
)


# ==============================
# PRODUCT NAME RESOLUTION
# ==============================

def resolve_product_name(product_name):
    """
    Map whatever the user typed onto a product that actually has
    reviews. "Lenovo Laptop" resolves to "Lenovo Laptop 1".
    Returns None when nothing matches.
    """

    wanted = str(product_name).strip().lower()

    if not wanted:
        return None

    names = reviews["ProductName"].astype(str)

    # Exact match
    exact = names[names.str.lower() == wanted]

    if not exact.empty:
        return exact.iloc[0]

    # Partial match - the user left off the product number
    partial = names[names.str.lower().str.contains(wanted, regex=False)]

    if not partial.empty:
        return sorted(partial.unique())[0]

    return None


# ==============================
# Q10 - ANALYZE REVIEW SENTIMENT
# ==============================

def analyze_reviews(product_name):

    resolved_name = resolve_product_name(product_name)

    if resolved_name is None:
        return None

    product_reviews = reviews[
        reviews["ProductName"].astype(str) == resolved_name
    ]

    review_texts = product_reviews["Review"].astype(str).tolist()

    if not review_texts:
        return None

    # Classify the whole batch in one call
    predictions = sentiment_analyzer(review_texts)

    positive = 0
    neutral = 0
    negative = 0

    detailed_reviews = []

    for text, result in zip(review_texts, predictions):

        label = result["label"]
        score = float(result["score"])

        if score < NEUTRAL_CONFIDENCE_THRESHOLD:
            sentiment = "Neutral"
            neutral += 1

        elif label == "POSITIVE":
            sentiment = "Positive"
            positive += 1

        else:
            sentiment = "Negative"
            negative += 1

        detailed_reviews.append({
            "Review": text,
            "Sentiment": sentiment,
            "Confidence": score
        })

    return {
        "product_name": resolved_name,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "total": len(review_texts),
        "details": detailed_reviews
    }


# ==============================
# Q11 - OLLAMA REVIEW SUMMARY
# ==============================

def generate_review_summary(product_name, review_data):

    review_text = ""

    for item in review_data["details"][:MAX_REVIEWS_IN_PROMPT]:

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
If there are no negative reviews, say so instead of inventing weaknesses.
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

def handle_review_question(product_name):

    review_data = analyze_reviews(product_name)

    if review_data is None:

        return {
            "found": False,
            "response": f"No reviews were found for {product_name}."
        }

    summary = generate_review_summary(
        review_data["product_name"],
        review_data
    )

    return {
        "found": True,
        "product_name": review_data["product_name"],
        "positive": review_data["positive"],
        "neutral": review_data["neutral"],
        "negative": review_data["negative"],
        "total": review_data["total"],
        "details": review_data["details"],
        "response": summary
    }
