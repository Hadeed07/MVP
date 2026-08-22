import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).resolve().parent / "owly.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_preferences (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            query TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()