import ollama


# ==============================
# INTENT DETECTION MODULE
# Q12 + WEB APPLICATION
# ==============================

def detect_intent(user_question):

    prompt = f"""
You are an intent classification system for an e-commerce chatbot.

Classify the user's question into EXACTLY ONE of these categories:

FAQ
RECOMMENDATION
PRODUCT
REVIEW

Definitions:

FAQ:
Questions about orders, delivery, tracking, cancellation, returns,
payment, shipping, and other frequently asked questions.

RECOMMENDATION:
Questions asking for product recommendations or suggestions.

PRODUCT:
Questions asking about a specific product, its price, brand,
category, description, or rating.

REVIEW:
Questions asking about customer reviews, opinions, sentiment,
whether a product is worth buying, strengths, weaknesses, or
customer satisfaction.

User Question:
{user_question}

IMPORTANT:
Return ONLY ONE WORD from:
FAQ
RECOMMENDATION
PRODUCT
REVIEW

Do not provide explanations.
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

    intent = response["message"]["content"].strip().upper()

    # Remove unwanted formatting
    intent = intent.replace(".", "").replace(":", "").strip()

    valid_intents = [
        "FAQ",
        "RECOMMENDATION",
        "PRODUCT",
        "REVIEW"
    ]

    if intent not in valid_intents:
        return "UNKNOWN"

    return intent