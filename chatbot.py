import sys

import pandas as pd


# ==============================
# E-COMMERCE AI CHATBOT
# Q1  - Dataset overview
# Q13 - Integrated chatbot
# ==============================

# Show every column when printing the datasets
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)

EXIT_COMMANDS = {"exit", "quit", "bye", "q"}

LINE = "=" * 60


# ==============================
# Q1 - LOAD AND DESCRIBE DATA
# ==============================

def show_dataset_overview():

    products = pd.read_csv("data/products.csv")
    faq = pd.read_csv("data/faq.csv")
    reviews = pd.read_csv("data/reviews.csv")

    datasets = [
        ("PRODUCTS", products),
        ("FAQ", faq),
        ("REVIEWS", reviews)
    ]

    for name, frame in datasets:

        print("\n" + LINE)
        print(name + " DATASET")
        print(LINE)

        print("\nFirst five records:")
        print(frame.head())

        rows, columns = frame.shape

        print("\nRows    :", rows)
        print("Columns :", columns)
        print("Attributes:", frame.columns.tolist())

    print()


# ==============================
# LOAD CHATBOT MODULES
# ==============================

def load_modules():
    """
    Imported inside a function because each module loads a model
    the first time it is imported, which takes a few seconds.
    """

    print("Loading Intent Module...")
    from intent_module import detect_intent

    print("Loading FAQ Module...")
    from faq_module import handle_faq

    print("Loading Recommendation Module...")
    from recommendation_module import handle_recommendation

    print("Loading Product Module...")
    from product_module import handle_product_question, find_product_by_name, products

    print("Loading Review Module...")
    from review_module import handle_review_question

    return {
        "detect_intent": detect_intent,
        "handle_faq": handle_faq,
        "handle_recommendation": handle_recommendation,
        "handle_product_question": handle_product_question,
        "find_product_by_name": find_product_by_name,
        "handle_review_question": handle_review_question,
        "products": products
    }


# ==============================
# INTENT HANDLERS
# ==============================

def run_faq(modules, question):
    """Q5 - answer from the FAQ, or report that nothing matched."""

    result = modules["handle_faq"](question)

    print("\nBest Matching FAQ :", result["question"])
    print("Similarity Score  :", result["similarity"])

    if not result["matched"]:
        return None

    return result["response"]


def run_recommendation(modules, question):
    """Q7 - retrieve top 3 products and recommend one."""

    result = modules["handle_recommendation"](question)

    if result["products"]:

        print("\nRetrieved Products")

        for product in result["products"]:
            print("   %-24s Rs %-10s Rating %s" % (
                product["ProductName"],
                int(product["Price"]),
                product["Rating"]
            ))

    return result["response"]


def run_product(modules, question, previous_product):
    """Q9 - answer a question about one product."""

    result = modules["handle_product_question"](
        question,
        previous_product
    )

    product = result["product"]

    if product is not None:
        print("\nRetrieved Product :", product["ProductName"])
        print("Similarity        :", result["similarity"])

    return result["response"], product


def run_review(modules, question, previous_product):
    """Q11 - summarise the reviews of one product."""

    # Which product are we talking about?
    index = modules["find_product_by_name"](question)

    if index is not None:
        product_name = modules["products"].iloc[index]["ProductName"]

    elif previous_product is not None:
        product_name = previous_product["ProductName"]

    else:
        return (
            "Please mention the product you want reviews for, "
            "for example \"What do customers say about Lenovo Laptop 3?\""
        )

    result = modules["handle_review_question"](product_name)

    if not result["found"]:
        return result["response"]

    print("\nReviews for       :", result["product_name"])
    print("Positive          :", result["positive"])
    print("Neutral           :", result["neutral"])
    print("Negative          :", result["negative"])

    print("\nDetailed Review")

    for item in result["details"]:
        print("   %-24s %-9s %.3f" % (
            item["Review"],
            item["Sentiment"],
            item["Confidence"]
        ))

    return result["response"]


# ==============================
# Q13 - ROUTE ONE QUESTION
# ==============================

def answer(modules, question, previous_product):

    # Q12 - decide which module should handle the question
    intent = modules["detect_intent"](question)

    print("\nDetected Intent   :", intent)

    if intent == "FAQ":

        response = run_faq(modules, question)

        if response is not None:
            return response, previous_product

        # Q5 - no FAQ was close enough, continue to the next module
        print("\nNo FAQ matched. Trying the product module...")
        intent = "PRODUCT"

    if intent == "RECOMMENDATION":
        return run_recommendation(modules, question), previous_product

    if intent == "PRODUCT":
        response, product = run_product(modules, question, previous_product)
        return response, product or previous_product

    if intent == "REVIEW":
        return run_review(modules, question, previous_product), previous_product

    return (
        "I'm not sure how to help with that. You can ask about "
        "orders, products, recommendations, or reviews."
    ), previous_product


# ==============================
# Q13 - CONTINUOUS INTERACTION
# ==============================

def chat_loop(modules):

    print("\n" + LINE)
    print("E-COMMERCE AI ASSISTANT")
    print(LINE)
    print("\nAsk about orders, products, recommendations or reviews.")
    print("Type 'exit' to quit.\n")

    # Remembers the product being discussed, so that follow-up
    # questions such as "What is its price?" keep working.
    previous_product = None

    while True:

        try:
            question = input("You : ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            return

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("\nGoodbye!")
            return

        try:
            response, previous_product = answer(
                modules,
                question,
                previous_product
            )

        except Exception as error:
            print("\nSomething went wrong:", error)
            print("Make sure Ollama is running (ollama serve).\n")
            continue

        print("\nAssistant")
        print(response)
        print("\n" + "-" * 60 + "\n")


# ==============================
# ENTRY POINT
# ==============================

def main():

    # python chatbot.py --data   -> Q1 dataset overview only
    if "--data" in sys.argv:
        show_dataset_overview()
        return

    modules = load_modules()

    chat_loop(modules)


if __name__ == "__main__":
    main()
