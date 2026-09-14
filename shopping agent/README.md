# 🛒 AI Shopping Assistant

A conversational shopping agent built with **LangChain**, **Groq**, and a **Streamlit** chat
UI. Describe (or photograph) a product and the agent searches a local catalog, checks
ratings, places orders, and remembers each user's order history and preferences across
sessions.

---

## Features

- **Keyword search** over the product catalog (name, description, category), with optional
  filters for maximum price and organic-only.
- **Ratings** — average customer rating and review count per product.
- **Checkout** — places an order and saves it to the database.
- **Shop by image** — upload a product photo; a vision model extracts its attributes and
  feeds them into search.
- **Order history** — answers "what have I ordered before?" per user.
- **User preferences** — remembers standing preferences (e.g. *always organic*, *never over
  $20*) so they don't have to be repeated each session.
- **Input guardrail** — off-topic messages (e.g. *"write me a poem"*, *"what's the weather?"*)
  are politely redirected before the agent runs, keeping the assistant focused on shopping.

---

## Project Structure

| File | Responsibility |
|------|----------------|
| `shopping_agent.py` | The LLM, the agent, its tools, and the system prompt. |
| `app.py` | Streamlit chat UI; holds chat state and the current user's identity. |
| `reviews_api.py` | Reads the `reviews` table and aggregates ratings. |
| `memory_setup.py` | One-time, idempotent database migration. |
| `store.db` | SQLite database (products, reviews, orders, preferences). |
| `resources/` | Sample product images for image search. |

---

## Database Schema

```sql
products(id, name, category, price, description, is_organic)
reviews(id, product_id, rating, reviewer_name, review_text)
orders(id, product_id, product_name, price, ordered_at, user_id)
preferences(user_id, key, value)   -- PRIMARY KEY (user_id, key)
```

`preferences` is a flexible key/value table, so new preference types never require a schema
change. Standard keys: `prefer_organic` (`"true"`/`"false"`) and `max_price` (a number as text).

---

## Setup

**Requirements:** Python 3.11+ and a [Groq API key](https://console.groq.com/).

Install dependencies:

```bash
pip install python-dotenv langchain langchain-core langchain-groq streamlit
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Run the one-time database migration (safe to re-run):

```bash
python memory_setup.py
```

---

## Running

Start the web app:

```bash
streamlit run app.py
```

Then in the browser:

1. Enter your name in the sidebar (**Your profile → Your name**).
2. Ask for what you want, e.g. *"I want organic honey under $20 with 4.5+ rating."*
3. Confirm an order by saying *"yes"* or *"order #2"*.
4. Ask *"what have I ordered before?"*, or set a preference like *"remember I always want organic."*

Command-line smoke test (no UI):

```bash
python shopping_agent.py
```

---

## Notes

- Each shopper is identified by a simple name typed in the sidebar (lowercased into a
  `user_id`). Orders and preferences are stored per user.
- The models are configured with `reasoning_effort="none"` and a modest `max_tokens` to stay
  within Groq's free-tier output-token-per-minute limit. On a paid tier these can be raised.
- The input guardrail is a lightweight one-word classification that runs *before* the agent, so
  off-topic messages are handled without a full multi-step agent run. It is applied to typed
  messages; image uploads are treated as shopping requests by design.
