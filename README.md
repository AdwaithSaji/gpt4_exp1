# E-Commerce AI Chatbot (RAG + Ollama)

**24CS2506 - Building Applications using GPT-4**
**Experiment 1 - Building an AI-Powered E-Commerce Chatbot using Ollama**

A modular Retrieval-Augmented Generation (RAG) chatbot for an online store.
It answers FAQs, recommends products, answers product-specific questions and
summarises customer reviews. A local LLM (Qwen, served by Ollama) both routes
the query and writes the final answer, while all facts come from the datasets.

---

## Workflow

```
User Query
    |
Intent Detection (Ollama)
    |
FAQ | RECOMMENDATION | PRODUCT | REVIEW
    |
Retrieval (Embeddings / TF-IDF / Reviews + Sentiment)
    |
Ollama (answer grounded in the retrieved context)
    |
Natural Language Response
```

The LLM is never asked to recall product facts from memory. It only rewrites
what retrieval hands it, which is what makes this a RAG system.

---

## Project structure

```
ECommerceGPT/
├── chatbot.py               # Q1 + Q13 - dataset overview and the CLI chatbot
├── app.py                   # Flask web version of the same chatbot
├── create_embeddings.py     # Q3 + Q6 - FAQ embeddings and product TF-IDF vectors
├── faq_module.py            # Q4 + Q5 - FAQ semantic search
├── recommendation_module.py # Q6 + Q7 - product recommendation
├── product_module.py        # Q8 + Q9 - product question answering (RAG)
├── review_module.py         # Q10 + Q11 - sentiment analysis and review summary
├── intent_module.py         # Q12 - LLM-based intent routing
├── data/                    # products.csv, faq.csv, reviews.csv
├── models/                  # generated embeddings and TF-IDF vectors
├── static/                  # style.css, script.js
└── templates/               # index.html
```

---

## Setup

**1. Install Ollama and pull the model**

Download Ollama from https://ollama.com/download, then:

```bash
ollama pull qwen2.5
ollama serve
```

**2. Install the Python dependencies**

```bash
pip install -r requirements.txt
```

**3. Build the embeddings and TF-IDF vectors**

```bash
python create_embeddings.py
```

This writes `models/faq_embeddings.pkl`, `models/tfidf_vectorizer.pkl` and
`models/product_vectors.pkl`. The modules rebuild them automatically if they
are missing, so this step is optional on a fresh clone.

---

## Running

Command-line chatbot (Q13):

```bash
python chatbot.py
```

Dataset overview only (Q1):

```bash
python chatbot.py --data
```

Web interface:

```bash
python app.py
```

Then open http://127.0.0.1:5000

---

## Part A - Dataset Preparation

### Q1. Load the datasets

`python chatbot.py --data` prints the first five records, the row and column
counts and the attribute list for each dataset.

| Dataset | Rows | Columns | Attributes |
|---|---|---|---|
| products.csv | 100 | 7 | ProductID, ProductName, Category, Brand, Price, Description, Rating |
| faq.csv | 20 | 3 | FAQID, Question, Answer |
| reviews.csv | 1000 | 5 | ReviewID, ProductID, ProductName, Review, Rating |

The catalogue covers 4 categories (Laptop, Mobile, Headphones, Smartwatch)
across 17 brands, with 10 reviews per product.

### Q2. Purpose of each dataset and how it maps to the chatbot modules

| Dataset | Purpose | Used by | Retrieval technique |
|---|---|---|---|
| **products.csv** | The product catalogue - name, brand, category, price, description and rating. Supplies every product fact the chatbot is allowed to state. | `recommendation_module.py`, `product_module.py` | TF-IDF vectors + cosine similarity |
| **faq.csv** | Canned answers to common customer-service questions (tracking, returns, cancellation, payment, delivery). Lets the bot answer policy questions without touching the catalogue. | `faq_module.py` | Sentence embeddings + cosine similarity |
| **reviews.csv** | Customer opinions per product. Turns raw opinion into a Positive / Neutral / Negative breakdown plus a buying suggestion. | `review_module.py` | Filter by product + transformer sentiment model |

`intent_module.py` sits in front of all three and decides which one a given
question belongs to.

---

## Part B - FAQ Semantic Search

- **Q3** `create_embeddings.build_faq_embeddings()` encodes all 20 FAQ
  questions with `all-MiniLM-L6-v2` and pickles the resulting matrix.
- **Q4** `faq_module.search_faq()` embeds the user question and takes the
  highest cosine similarity against the stored FAQ vectors.
- **Q5** `faq_module.handle_faq()` applies `SIMILARITY_THRESHOLD = 0.50`.
  Above it, Ollama rewrites the stored answer as a natural sentence. Below it
  the FAQ is treated as a miss and the chatbot falls through to the product
  module rather than answering from an unrelated FAQ.

```
You : Where is my parcel?

Detected Intent   : FAQ
Best Matching FAQ : How do I track my order?
Similarity Score  : 0.94

Assistant
You can track your order from the Track Order page using your order ID.
```

Semantic search is what makes this work: "Where is my parcel?" shares almost
no words with "How do I track my order?", but the two sentences sit close
together in embedding space.

---

## Part C - Product Recommendation

- **Q6** `create_embeddings.build_product_vectors()` fits a TF-IDF vectorizer
  over the product text and stores both the vectorizer and the document matrix.
- **Q7** `recommendation_module.handle_recommendation()` retrieves the top 3
  products by cosine similarity and asks Ollama to pick the best one and
  justify it from the retrieved fields only.

```
You : Recommend a gaming laptop

Detected Intent   : RECOMMENDATION

Retrieved Products
   ASUS Laptop 3            Rs 29108      Rating 4.9
   HP Laptop 4              Rs 16717      Rating 4.8
   Dell Laptop 1            Rs 80726      Rating 4.8

Assistant
I recommend ASUS Laptop 3.
Brand  : ASUS
Price  : Rs 29,108
Reason : It is the highest rated laptop among the retrieved matches
         and offers a good balance of performance and value.
```

**Two retrieval decisions worth noting.** Every description in this dataset is
generated from one template - *"Laptop by Lenovo suitable for everyday use with
quality performance."* Vectorising the description alone therefore gives every
laptop an identical vector, and the "top 3" becomes whichever rows happen to
sort last. Two changes fix that:

1. The TF-IDF corpus combines **ProductName + Category + Brand + Description**,
   so brand and category queries actually discriminate. The vectorizer also
   keeps single-character tokens, so the trailing product number survives - the
   default token pattern drops it, which is why "Lenovo Laptop 3" used to be
   indistinguishable from "Lenovo Laptop 2".
2. Exact similarity ties are broken by **customer rating**, so the top 3 are the
   best-reviewed matches instead of an arbitrary three.

Queries whose terms appear nowhere in the catalogue return a "no match" message
instead of three unrelated products.

---

## Part D - Product Question Answering (RAG)

- **Q8** `product_module.retrieve_product()` first looks for a product named
  explicitly in the question, and otherwise falls back to TF-IDF retrieval.
  The chosen product is turned into a structured context block.
- **Q9** `product_module.answer_product_question()` sends that context to
  Ollama with a constrained prompt: answer only from the context, invent
  nothing, and say so when a detail is absent.

```
You : Tell me about Lenovo Laptop 3

Detected Intent   : PRODUCT
Retrieved Product : Lenovo Laptop 3
Similarity        : 1.0

Assistant
Lenovo Laptop 3 is a laptop manufactured by Lenovo, priced at Rs 7,081,
with a customer rating of 4.3.

You : What is its price?

Assistant
The price of Lenovo Laptop 3 is Rs 7,081.
```

**Conversation memory.** The chatbot remembers the product under discussion so
that "What is its price?" resolves correctly. The follow-up test is deliberately
narrow: a question counts as a follow-up only when it carries a pronoun or a
bare attribute *and* names no brand or category of its own. Without that second
condition, *"What is the price of Dell Laptop 5?"* would be treated as a
follow-up merely because it contains the word "price", and would be answered
about the previous product.

---

## Part E - Customer Review Intelligence

- **Q10** `review_module.py` loads the pre-trained
  `distilbert-base-uncased-finetuned-sst-2-english` sentiment model.
- **Q11** `handle_review_question()` classifies every review for the product
  and asks Ollama for a structured summary: Overall Opinion, Strengths,
  Weaknesses and a Buying Suggestion.

```
You : What do customers say about Lenovo Laptop 3?

Detected Intent   : REVIEW
Reviews for       : Lenovo Laptop 3
Positive          : 7
Neutral           : 1
Negative          : 2

Detailed Review
   Worth the money          Positive  0.998
   Average quality          Neutral   0.720
   Poor battery             Negative  0.995
   ...
```

**Where Neutral comes from.** SST-2 is a binary model - it only ever emits
POSITIVE or NEGATIVE, so a naive mapping can never produce the Neutral bucket
the experiment asks for. Reviews such as *"Average quality"* land near the
decision boundary, so anything the model is not confident about is reported as
Neutral. The cut-off is `NEUTRAL_CONFIDENCE_THRESHOLD = 0.90` in
`review_module.py` and can be tuned.

Product names are resolved loosely, so "Lenovo Laptop" finds "Lenovo Laptop 1"
rather than returning nothing.

---

## Part F - Intelligent Decision Making

- **Q12** `intent_module.detect_intent()` asks Ollama to classify the question
  into FAQ, RECOMMENDATION, PRODUCT or REVIEW.

| Question | Intent |
|---|---|
| Where is my parcel? | FAQ |
| Recommend a gaming laptop | RECOMMENDATION |
| Tell me about Samsung Galaxy | PRODUCT |
| What do customers say about Lenovo Laptop 3? | REVIEW |

Small models rarely return a bare word - they answer `**FAQ**`, `Intent: FAQ`
or `FAQ.` instead. `parse_intent()` extracts the first valid label from the
reply rather than comparing it literally, and returns `UNKNOWN` if none is
present.

- **Q13** `chatbot.py` wires everything together: it detects the intent, routes
  to the matching module, keeps the current product in memory for follow-up
  questions, recovers from errors without dropping the session, and loops until
  the user types `exit`.

---

## Concepts used

| Concept | Where it appears |
|---|---|
| **Embeddings** - text as numeric vectors | FAQ questions encoded with all-MiniLM-L6-v2 |
| **Semantic search** - matching meaning, not words | "Where is my parcel?" finds "How do I track my order?" |
| **Cosine similarity** - 1 identical, 0 unrelated | FAQ matching, product retrieval, recommendation ranking |
| **TF-IDF** - term importance weighting | Product catalogue vectors |
| **Sentiment analysis** - opinion polarity | DistilBERT SST-2 over customer reviews |
| **RAG** - retrieve, then generate | Every module grounds Ollama in retrieved context |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ConnectionError` or "Make sure Ollama is running" | Start the server with `ollama serve` |
| `model 'qwen2.5' not found` | Run `ollama pull qwen2.5` |
| `FileNotFoundError` on a `.pkl` file | Run `python create_embeddings.py` |
| First run is slow | The embedding and sentiment models download once, then cache |
