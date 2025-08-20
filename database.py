import sqlite3
import json

DB_NAME = "dashboards.db"

def initialize_db():
    """Initializes the database and creates the table if it doesn't exist."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_dashboards (
                username TEXT PRIMARY KEY,
                dashboard_data TEXT
            )
        """)
        conn.commit()

def save_dashboard_to_db(username, dashboard_data):
    """Saves or updates a user's dashboard data in the database."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        # Use INSERT OR REPLACE to handle both new and existing users
        cursor.execute("""
            INSERT OR REPLACE INTO user_dashboards (username, dashboard_data)
            VALUES (?, ?)
        """, (username, json.dumps(dashboard_data)))
        conn.commit()

def load_dashboard_from_db(username):
    """Loads a user's dashboard data from the database."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT dashboard_data FROM user_dashboards WHERE username = ?", (username,))
        result = cursor.fetchone()
        return json.loads(result[0]) if result else []