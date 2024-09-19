import base64
import aiosqlite
from quart import g, current_app
import logging
import json
import asyncio

class SQLiteManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

    async def optimize_sqlite(self):
        """Optimize SQLite settings for better performance."""
        conn = await self.get_connection()
        await conn.execute("PRAGMA synchronous = OFF")
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA page_size = 4096")
        await conn.commit()

    async def get_connection(self):
        if current_app:
            if 'db_connection' not in g:
                g.db_connection = await aiosqlite.connect(self.db_path)
                g.db_connection.row_factory = aiosqlite.Row
            return g.db_connection
        else:
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            return conn

    async def close_connection(self, error=None):
        conn = g.pop('db_connection', None)
        if conn:
            await conn.close()

    async def create_table(self):
        """Create necessary tables if they don't exist."""
        conn = await self.get_connection()
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                front_side_ocr TEXT,
                front_side_qr TEXT,
                back_side_ocr TEXT,
                back_side_qr TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS user_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ocr_result_id INTEGER,
                context TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ocr_result_id) REFERENCES ocr_results(id)
            )
        ''')
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ocr_result_id INTEGER,
                image_type TEXT NOT NULL,
                image_data TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ocr_result_id) REFERENCES ocr_results(id)
            )
        ''')
        await conn.commit()

    async def insert_scan_result(self, data: dict):
        """Insert scan result into the scan_results table."""
        conn = await self.get_connection()
        await conn.execute('INSERT INTO scan_results (result_json) VALUES (?)', (json.dumps(data),))
        await conn.commit()

    async def insert_ocr_result(self, data: dict) -> int:
        import json  # Ensure this import is present
        conn = await self.get_connection()
        cursor = await conn.execute('''
            INSERT INTO ocr_results (
                front_side_ocr,
                front_side_qr,
                back_side_ocr,
                back_side_qr
            ) VALUES (?, ?, ?, ?)
        ''', (
            json.dumps(data.get('front_side_ocr')),  # Serialize to JSON string
            data.get('front_side_qr'),
            json.dumps(data.get('back_side_ocr')),   # Serialize to JSON string
            data.get('back_side_qr')
        ))
        await conn.commit()
        return cursor.lastrowid


    async def insert_user_context(self, ocr_result_id: int, context: str):
        """Insert generated user context into the user_contexts table."""
        conn = await self.get_connection()
        await conn.execute('''
            INSERT INTO user_contexts (ocr_result_id, context)
            VALUES (?, ?)
        ''', (ocr_result_id, context))
        await conn.commit()

    async def insert_image(self, ocr_result_id: int, image_type: str, image_data: str):
        """Insert base64 encoded image data into the images table."""
        conn = await self.get_connection()
        await conn.execute('''
            INSERT INTO images (ocr_result_id, image_type, image_data)
            VALUES (?, ?, ?)
        ''', (ocr_result_id, image_type, image_data))
        await conn.commit()

    @staticmethod
    async def image_to_base64(image_path: str) -> str:
        """Convert image file to base64 string."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, SQLiteManager._sync_image_to_base64, image_path)

    @staticmethod
    def _sync_image_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
