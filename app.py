from flask import Flask, render_template, request, jsonify

from intent_module import detect_intent
from faq_module import handle_faq
from recommendation_module import handle_recommendation
from product_module import handle_product_question
from review_module import handle_review_question


# ==============================
# FLASK APPLICATION
# ==============================

app = Flask(__name__)


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# CHAT API
# ==============================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_question = data.get("message", "").strip()

    # Product remembered from previous conversation
    previous_product = data.get("previous_product")

    if not user_question:

        return jsonify({
            "success": False,
            "message": "Please enter a question."
        })


    try:

        # ==============================
        # STEP 1 - INTENT DETECTION
        # ==============================

        intent = detect_intent(user_question)


        # ==============================
        # STEP 2 - ROUTE TO MODULE
        # ==============================

        if intent == "FAQ":

            result = handle_faq(user_question)

            # If FAQ similarity is too low
            if not result["matched"]:

                return jsonify({
                    "success": True,
                    "intent": "FAQ",
                    "response": "I couldn't find a suitable answer in the FAQ. Please try asking your question in a different way.",
                    "previous_product": previous_product
                })

            return jsonify({
                "success": True,
                "intent": "FAQ",
                "response": result["response"],
                "similarity": result["similarity"],
                "previous_product": previous_product
            })


        # ==============================
        # RECOMMENDATION
        # ==============================

        elif intent == "RECOMMENDATION":

            result = handle_recommendation(
                user_question
            )

            return jsonify({
                "success": True,
                "intent": "RECOMMENDATION",
                "response": result["response"],
                "products": result["products"],
                "previous_product": previous_product
            })


        # ==============================
        # PRODUCT QUESTION
        # ==============================

        elif intent == "PRODUCT":

            result = handle_product_question(
                user_question,
                previous_product
            )

            product = result["product"]

            return jsonify({
                "success": True,
                "intent": "PRODUCT",
                "response": result["response"],
                "product": product,
                "similarity": result["similarity"],
                "previous_product": product
            })


        # ==============================
        # REVIEW
        # ==============================

        elif intent == "REVIEW":

            # If a product was already selected,
            # use it for follow-up review questions.

            if previous_product:

                product_name = previous_product["ProductName"]

            else:

                # Try to find the product name from
                # the user's question.

                import pandas as pd

                products = pd.read_csv(
                    "data/products.csv"
                )

                product_name = None

                question_lower = user_question.lower()

                for name in products["ProductName"]:

                    if str(name).lower() in question_lower:

                        product_name = name
                        break

                if product_name is None:

                    return jsonify({
                        "success": True,
                        "intent": "REVIEW",
                        "response": "Please mention the product name you want to know about.",
                        "previous_product": previous_product
                    })


            result = handle_review_question(
                product_name
            )

            return jsonify({
                "success": True,
                "intent": "REVIEW",
                "response": result["response"],
                "positive": result.get("positive", 0),
                "neutral": result.get("neutral", 0),
                "negative": result.get("negative", 0),
                "previous_product": previous_product
            })


        # ==============================
        # UNKNOWN INTENT
        # ==============================

        else:

            return jsonify({
                "success": True,
                "intent": "UNKNOWN",
                "response": "I'm not sure how to help with that. You can ask about orders, products, recommendations, or reviews.",
                "previous_product": previous_product
            })


    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "success": False,
            "message": "Something went wrong while processing your request.",
            "error": str(e)
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