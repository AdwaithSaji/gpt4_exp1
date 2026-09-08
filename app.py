from flask import Flask, render_template, request, jsonify

from intent_module import detect_intent
from faq_module import handle_faq
from recommendation_module import handle_recommendation
from product_module import (
    handle_product_question,
    find_product_by_name,
    products
)
from review_module import handle_review_question


# ==============================
# FLASK APPLICATION
# Web front-end for the chatbot
# ==============================

app = Flask(__name__)


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# INTENT HANDLERS
# ==============================

def faq_reply(question, previous_product):

    result = handle_faq(question)

    if not result["matched"]:
        return None

    return {
        "success": True,
        "intent": "FAQ",
        "response": result["response"],
        "matched_question": result["question"],
        "similarity": result["similarity"],
        "previous_product": previous_product
    }


def recommendation_reply(question, previous_product):

    result = handle_recommendation(question)

    return {
        "success": True,
        "intent": "RECOMMENDATION",
        "response": result["response"],
        "products": result["products"],
        "previous_product": previous_product
    }


def product_reply(question, previous_product):

    result = handle_product_question(question, previous_product)

    product = result["product"]

    return {
        "success": True,
        "intent": "PRODUCT",
        "response": result["response"],
        "product": product,
        "similarity": result["similarity"],
        # Remember the product for follow-up questions
        "previous_product": product or previous_product
    }


def review_reply(question, previous_product):

    # Work out which product the user means
    index = find_product_by_name(question)

    if index is not None:
        product_name = products.iloc[index]["ProductName"]

    elif previous_product:
        product_name = previous_product["ProductName"]

    else:
        return {
            "success": True,
            "intent": "REVIEW",
            "response": (
                "Please mention the product you want reviews for, "
                "for example \"What do customers say about Lenovo Laptop 3?\""
            ),
            "previous_product": previous_product
        }

    result = handle_review_question(product_name)

    return {
        "success": True,
        "intent": "REVIEW",
        "response": result["response"],
        "product_name": result.get("product_name", product_name),
        "positive": result.get("positive", 0),
        "neutral": result.get("neutral", 0),
        "negative": result.get("negative", 0),
        "details": result.get("details", []),
        "previous_product": previous_product
    }


# ==============================
# CHAT API
# ==============================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}

    user_question = str(data.get("message", "")).strip()

    # Product remembered from the previous turn
    previous_product = data.get("previous_product")

    if not user_question:

        return jsonify({
            "success": False,
            "message": "Please enter a question."
        })

    try:

        # STEP 1 - intent detection
        intent = detect_intent(user_question)

        # STEP 2 - route to the matching module
        if intent == "FAQ":

            reply = faq_reply(user_question, previous_product)

            if reply is not None:
                return jsonify(reply)

            # Nothing in the FAQ was close enough,
            # continue to the product module.
            intent = "PRODUCT"

        if intent == "RECOMMENDATION":
            return jsonify(
                recommendation_reply(user_question, previous_product)
            )

        if intent == "PRODUCT":
            return jsonify(
                product_reply(user_question, previous_product)
            )

        if intent == "REVIEW":
            return jsonify(
                review_reply(user_question, previous_product)
            )

        return jsonify({
            "success": True,
            "intent": "UNKNOWN",
            "response": (
                "I'm not sure how to help with that. You can ask about "
                "orders, products, recommendations, or reviews."
            ),
            "previous_product": previous_product
        })

    except Exception as error:

        print("ERROR:", error)

        return jsonify({
            "success": False,
            "message": (
                "Something went wrong while processing your request. "
                "Make sure Ollama is running (ollama serve)."
            ),
            "error": str(error)
        })


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
