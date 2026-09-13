"""
One-time (idempotent) setup for the agent's memory features.

Run this once to upgrade store.db so it can remember things per user:
    python memory_setup.py

It is safe to run repeatedly — it only makes changes that are missing.
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")


def setup_memory() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Add a user_id column to `orders` so each order belongs to a person.
    #    Existing orders (placed before we had users) default to 'default_user'.
    existing_columns = [row[1] for row in cursor.execute("PRAGMA table_info(orders)")]
    if "user_id" not in existing_columns:
        cursor.execute(
            "ALTER TABLE orders ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'"
        )
        print("Added 'user_id' column to orders.")
    else:
        print("orders.user_id already exists — skipping.")

    # 2. Create a flexible key/value preferences table.
    #    One row per (user, preference), e.g. ('alice', 'prefer_organic', 'true').
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS preferences (
            user_id TEXT NOT NULL,
            key     TEXT NOT NULL,
            value   TEXT,
            PRIMARY KEY (user_id, key)
        )
        """
    )
    print("preferences table is ready.")

    conn.commit()
    conn.close()
    print("Memory setup complete.")


if __name__ == "__main__":
    setup_memory()
