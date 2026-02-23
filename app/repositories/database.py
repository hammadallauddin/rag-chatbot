"""Database repository for managing SQLite operations."""
import sqlite3
from contextlib import contextmanager
from typing import Generator, Any
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseRepository:
    """Repository for database operations."""

    def __init__(self, db_name: str = None):
        """Initialize database with the given database name."""
        self.db_name = db_name or settings.db_name

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def initialize_tables(self) -> None:
        """Create required database tables if they don't exist."""
        with self._get_connection() as conn:
            # Application logs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS application_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_query TEXT,
                    gpt_response TEXT,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Document store table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Database tables initialized successfully")

    def insert_chat_log(
        self,
        session_id: str,
        user_query: str,
        gpt_response: str,
        model: str
    ) -> None:
        """Insert a chat log entry."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)",
                (session_id, user_query, gpt_response, model)
            )
            conn.commit()

    def get_chat_history(self, session_id: str) -> list[dict[str, str]]:
        """Retrieve chat history for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            )
            messages = []
            for row in cursor.fetchall():
                messages.extend([
                    {"role": "human", "content": row["user_query"]},
                    {"role": "ai", "content": row["gpt_response"]}
                ])
            return messages

    def insert_document_record(self, filename: str) -> int:
        """Insert a document record and return the file ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO document_store (filename) VALUES (?)",
                (filename,)
            )
            conn.commit()
            return cursor.lastrowid

    def delete_document_record(self, file_id: int) -> bool:
        """Delete a document record."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM document_store WHERE id = ?", (file_id,))
            conn.commit()
            return True

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Get all document records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC"
            )
            documents = cursor.fetchall()
            return [dict(doc) for doc in documents]


# Singleton instance
db_repository = DatabaseRepository()
