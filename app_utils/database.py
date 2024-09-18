import sqlite3
from flask import g, current_app
from typing import List, Dict, Any, Optional

class SQLiteManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
    def optimize_sqlite(self):
        """Optimize SQLite settings for better performance."""
        with self.get_connection() as conn:
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA page_size = 4096")  # Adjust page size as needed

    def get_connection(self):
        if current_app:
            # Use Flask's `g` to manage the connection in the app context
            if 'db_connection' not in g:
                g.db_connection = sqlite3.connect(self.db_path)
                g.db_connection.row_factory = sqlite3.Row  # Enables dict-like cursor
            return g.db_connection
        else:
            # Use thread-local storage for connections outside Flask context
            if not hasattr(self.local, 'connection'):
                self.local.connection = sqlite3.connect(self.db_path)
                self.local.connection.row_factory = sqlite3.Row
            return self.local.connection

    def close_connection(self, error=None):
        if current_app and 'db_connection' in g:
            g.db_connection.close()
            g.pop('db_connection', None)
        elif hasattr(self.local, 'connection'):
            self.local.connection.close()
            del self.local.connection

    def create_table(self):
        self.execute_with_timeout(self._create_table)

    def _create_table(self):
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(uid, embedding)
                )
            """)


    def _create_task_status_table(self):
        with self.transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL UNIQUE,
                    uid TEXT,
                    status TEXT NOT NULL,
                    is_final_task BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_uid ON task_status (uid)
            """)

    def execute_raw_sql(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        return self.execute_with_timeout(self._execute_raw_sql, args=(query, params))

    def _execute_raw_sql(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def transaction(self):
        """Context manager for transaction management."""
        class TransactionContext:
            def __init__(self, manager):
                self.manager = manager
                self.conn = None

            def __enter__(self):
                self.conn = self.manager.get_connection()
                self.manager.logger.info("Starting a new transaction...")
                self.conn.execute("BEGIN TRANSACTION")
                return self.conn

            def __exit__(self, exc_type, exc_value, traceback):
                if exc_type is None:
                    try:
                        self.manager.logger.info("Committing the transaction...")
                        self.conn.execute("COMMIT")
                    except sqlite3.Error as e:
                        self.manager.logger.error(f"Failed to commit transaction: {e}")
                        self.conn.execute("ROLLBACK")
                        raise
                else:
                    self.manager.logger.info("Rolling back the transaction due to an exception...")
                    self.conn.execute("ROLLBACK")

        return TransactionContext(self)

    def close(self):
        self.close_connection()
