import re

import ollama


# ==============================
# INTENT DETECTION MODULE
# Q12 + WEB APPLICATION
# ==============================

OLLAMA_MODEL = "qwen2.5"

VALID_INTENTS = [
    "FAQ",
    "RECOMMENDATION",
    "PRODUCT",
    "REVIEW"
]


PROMPT_TEMPLATE = """
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

Examples:

Where is my parcel? -> FAQ
Recommend a gaming laptop -> RECOMMENDATION
Tell me about Samsung Galaxy -> PRODUCT
What do customers say about Lenovo Laptop 3? -> REVIEW

User Question:
{question}

IMPORTANT:
Return ONLY ONE WORD from:
FAQ
RECOMMENDATION
PRODUCT
REVIEW

Do not provide explanations.
"""


# ==============================
# CLEAN THE MODEL OUTPUT
# ==============================

def parse_intent(raw_text):
    """
    Pull a valid intent out of the model's reply.

    Small models often answer with extra formatting such as
    "**FAQ**", "Intent: FAQ" or "FAQ." rather than a bare word,
    so the label is extracted instead of compared literally.
    """

    text = str(raw_text).upper()

    # Keep letters only, so markdown and punctuation fall away
    words = re.findall(r"[A-Z]+", text)

    for word in words:
        if word in VALID_INTENTS:
            return word

    return "UNKNOWN"


# ==============================
# Q12 - DETECT INTENT
# ==============================

def detect_intent(user_question):

    prompt = PROMPT_TEMPLATE.format(question=user_question)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return parse_intent(response["message"]["content"])
