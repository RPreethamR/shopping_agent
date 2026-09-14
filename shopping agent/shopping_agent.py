import base64
import json
import os
import sqlite3
from typing import Optional, Union

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

# reasoning_effort="none" turns off the model's hidden "thinking" tokens (it still
# calls tools normally). Without this, a reasoning model can spend its whole output
# budget thinking and finish with an empty reply. It also keeps each step small, which
# matters on Groq's free tier (limited output tokens per minute). max_tokens then caps
# the visible answer length.
llm = ChatGroq(
    model="qwen/qwen3.6-27b", temperature=0, max_tokens=512, reasoning_effort="none"
)
vision_llm = ChatGroq(
    model="qwen/qwen3.6-27b", temperature=0, max_tokens=512, reasoning_effort="none"
)
# A tiny model used only by the input guardrail. It answers a single word
# (SHOPPING / OFF_TOPIC), so max_tokens is very small to stay cheap.
guardrail_llm = ChatGroq(
    model="qwen/qwen3.6-27b", temperature=0, max_tokens=8, reasoning_effort="none"
)


# ---------------------------------------------------------------------------
# Helpers — some models (e.g. qwen on Groq) emit booleans/numbers as strings
# like "True" or "20". These normalize such values so tool calls don't fail
# schema validation.
# ---------------------------------------------------------------------------

def _coerce_bool(value) -> Optional[bool]:
    """Normalize a bool-ish value to True/False, or None if absent/unclear."""
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "organic"):
        return True
    if text in ("false", "0", "no", "non-organic", "nonorganic", "not organic"):
        return False
    return None  # unrecognized -> treat as no filter


def _coerce_float(value) -> Optional[float]:
    """Normalize a number that may arrive as a string (e.g. '20') to float, or None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Input guardrail — runs BEFORE the agent to reject off-topic messages
# (e.g. "write me a poem", "what's the weather") with a polite redirect, so we
# don't spend a full multi-step agent turn on something unrelated to shopping.
# ---------------------------------------------------------------------------

GUARDRAIL_REDIRECT = (
    "I'm your shopping assistant, so I can only help with shopping — finding "
    "products, checking ratings, placing orders, and your order history or "
    "preferences. What would you like to shop for today?"
)

# Common one-word/short confirmations we let straight through without a model
# call. In an ongoing chat these are follow-ups (e.g. answering "order it?"),
# and classifying them in isolation would be wasteful and error-prone.
_FOLLOWUP_OK = {
    "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "no", "nope",
    "confirm", "buy it", "order it", "go ahead", "sounds good",
}


def check_input_guardrail(message: str) -> Optional[str]:
    """Decide whether `message` is shopping-related.

    Returns None if the message is on-topic (let the agent handle it), or a
    polite redirect string if it is off-topic (skip the agent). Uses one cheap
    one-word LLM classification. Fails OPEN: if the verdict is unclear, we allow
    the message through rather than wrongly blocking a real request.
    """
    text = (message or "").strip()
    if not text:
        return None

    # Fast path: short confirmations/replies that continue a shopping chat.
    normalized = text.lower().rstrip(".!?")
    if normalized in _FOLLOWUP_OK or normalized.startswith(("#", "order #")):
        return None

    system = (
        "You are a strict input filter for an online grocery/product shopping "
        "assistant. Decide if the user's message belongs to shopping: searching "
        "for products, prices, ratings, placing or reviewing orders, order "
        "history, or shopping preferences. Greetings and brief replies that could "
        "continue a shopping chat count as shopping. Anything unrelated (poems, "
        "weather, coding help, general trivia, etc.) is off-topic.\n"
        "Answer with EXACTLY one word: SHOPPING or OFF_TOPIC."
    )
    response = guardrail_llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
    )
    verdict = (response.content or "").strip().upper()

    # Fail open: only block when the model clearly says OFF_TOPIC.
    if "OFF_TOPIC" in verdict:
        return GUARDRAIL_REDIRECT
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(
    query: str,
    max_price: Optional[Union[float, str]] = None,
    is_organic: Optional[Union[bool, str]] = None,
) -> str:
    """
    Search the product database by keyword (matched against name, description, and category).
    Optionally filter by maximum price and/or organic status.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic.
    """
    max_price = _coerce_float(max_price)
    is_organic = _coerce_bool(is_organic)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params: list = []

    if query:
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    products = [
        {
            "id":          row[0],
            "name":        row[1],
            "category":    row[2],
            "price":       row[3],
            "description": row[4],
            "is_organic":  bool(row[5]),
        }
        for row in rows
    ]
    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id)
    return json.dumps(result)


@tool
def checkout(product_id: int, user_id: str) -> str:
    """
    Place an order for the given product ID on behalf of user_id. Saves the order to the
    database (attributed to that user) and returns a confirmation message with the order
    ID, product name, and price.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price, user_id) VALUES (?, ?, ?, ?)",
        (product_id, name, price, user_id),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )


@tool
def get_order_history(user_id: str) -> str:
    """
    Get the past orders placed by user_id (most recent first). Use this to answer
    questions like "what have I ordered before?" or "what did I buy last time?".
    Returns a JSON array of orders, each with: order_id, product_name, price, ordered_at.
    An empty array means the user has not ordered anything yet.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, product_name, price, ordered_at FROM orders "
        "WHERE user_id = ? ORDER BY ordered_at DESC",
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    orders = [
        {
            "order_id":     row[0],
            "product_name": row[1],
            "price":        row[2],
            "ordered_at":   row[3],
        }
        for row in rows
    ]
    return json.dumps(orders)


@tool
def get_preferences(user_id: str) -> str:
    """
    Get the saved shopping preferences for user_id. Call this at the start of a shopping
    request so you can apply the user's standing preferences without asking them again.
    Returns a JSON object of key/value pairs. Recognized keys:
    - prefer_organic: "true" or "false" — whether the user wants organic items by default.
    - max_price: a number as text (e.g. "20") — the user never wants items priced above this.
    An empty object ({}) means the user has no saved preferences yet.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM preferences WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return json.dumps({row[0]: row[1] for row in rows})


@tool
def save_preference(user_id: str, key: str, value: str) -> str:
    """
    Save (or update) one standing preference for user_id so it is remembered across
    sessions. Call this only when the user expresses a LASTING preference, e.g. "I always
    want organic" or "never show me anything over $20" — not for a one-off request.
    Use these standard keys:
    - prefer_organic: "true" or "false"
    - max_price: a number as text, e.g. "20"
    Returns a short confirmation message.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO preferences (user_id, key, value) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
        (user_id, key, value),
    )
    conn.commit()
    conn.close()
    return f"Saved preference: {key} = {value}."


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with search_products.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                "Look at this product image and extract its key attributes. "
                "Return ONLY a JSON object with these fields:\n"
                "- product_type: what kind of product it is (e.g. honey, olive oil, almonds)\n"
                "- search_query: a short keyword to search for it (e.g. 'honey', 'olive oil')\n"
                "- is_organic: true if the label says organic, false if not, null if unclear\n"
                "- description: one sentence describing the product"
            ),
        },
    ])

    response = vision_llm.invoke([message])
    return response.content


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_agent(
    tools=[
        search_products, get_rating, checkout, get_order_history,
        get_preferences, save_preference, describe_product_image,
    ],
    model=llm,
    system_prompt=(
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
        "USER IDENTITY — the current user's user_id is provided to you in a system message "
        "at the start of the conversation. Whenever a tool asks for user_id (checkout, "
        "get_order_history, get_preferences, save_preference), always pass that exact "
        "user_id. Never invent or guess a user_id.\n\n"
        "PREFERENCES — the user has standing preferences you should honor automatically so "
        "they don't have to repeat them:\n"
        "1. The FIRST time the user asks to browse or buy in a conversation, call "
        "   get_preferences with the current user_id.\n"
        "2. Apply saved preferences to search_products, unless the user overrides them this "
        "   turn:\n"
        "     - prefer_organic = 'true'  -> pass is_organic=true\n"
        "     - max_price = <number>     -> pass max_price=<number>\n"
        "   When you apply a saved preference, briefly mention it (e.g. 'Applying your usual "
        "   organic-only filter and $20 cap.').\n"
        "3. When the user states a LASTING preference (e.g. 'I always want organic', 'never "
        "   show me anything over $20', 'remember I prefer organic'), call save_preference to "
        "   store it (prefer_organic / max_price), then confirm what you saved.\n"
        "4. A one-off request for the current search only (e.g. 'just this time show me "
        "   non-organic') must NOT be saved.\n\n"
        "ORDER HISTORY — when the user asks about their past orders (e.g. 'what have I ordered "
        "before?', 'what did I buy last time?', 'show my order history'):\n"
        "1. Call get_order_history with the current user_id.\n"
        "2. If the array is empty, tell them they haven't placed any orders yet.\n"
        "3. Otherwise summarize their orders in plain text — one line per order with the "
        "   product name, price, and date. Do not use backticks, bold, or code blocks.\n\n"
        "IMAGE SEARCH — when the user provides an image path:\n"
        "1. Call describe_product_image with the path to identify the product.\n"
        "2. Use the returned search_query and is_organic to call search_products.\n"
        "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
        "BROWSING — when the user describes what they want to buy:\n"
        "1. Call search_products to find matching items, applying any price/organic filters "
        "   the user gave this turn AND their saved preferences (see PREFERENCES above).\n"
        "2. For each candidate, call get_rating to retrieve its average rating.\n"
        "3. Filter by the user's minimum rating if specified.\n"
        "4. Present qualifying products as a numbered list. For each item use this exact format "
        "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
        "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
        "   Add a blank line between each product entry for readability. "
        "   Always include (ID:X) so you can reference it later.\n"
        "5. If only one product qualifies, still show it in the list and ask: "
        "   'Would you like to order it? Just say yes or give me the number.'\n"
        "6. Do NOT call checkout at this stage.\n\n"
        "ORDERING — when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
        "'order number 2', 'the first one', 'get me #3'):\n"
        "1. Look at your previous message to find the (ID:X) for the chosen product "
        "   (if only one was listed and the user says 'yes', use that product's ID).\n"
        "2. You MUST call the checkout tool with that product_id (the number from (ID:X)) and "
        "   the current user_id. The order is NOT placed until this tool call runs — writing a "
        "   confirmation message yourself does NOT create an order.\n"
        "3. Only AFTER checkout returns, confirm to the user using the exact order ID and "
        "   details from the tool's response. NEVER make up an order ID, and NEVER claim an "
        "   order succeeded unless the checkout tool actually returned a success message.\n\n"
        "Never place an order unless the user explicitly confirms. "
        "Never guess a product_id — always take it from the (ID:X) in your own previous message."
    ),
)

if __name__ == "__main__":
    demo_user = "demo_user"
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"The current user's user_id is '{demo_user}'. "
                        "Use this exact user_id for any tool that needs one."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "I want to buy organic honey with 4.5+ rating and less than $20 price."
                    ),
                },
            ]
        }
    )
    print(result["messages"][-1].content)
