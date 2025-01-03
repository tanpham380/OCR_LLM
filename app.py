# app.py
from quart import Quart, g
from app_utils.database import SQLiteManager
from app_utils.logging import get_logger
from app_utils.routes import blueprint  # Adjust the import path as necessary
import os

from config import (
    BASE_PATH,
    LOG_PATH_FILE,
    SAVE_DIR,
    SAVE_IMAGES,
    SAVE_RESULT,
    UTILS_PATH,
)

# Initialize logger
logger = get_logger(__name__)

# Define base paths
directories = [
    LOG_PATH_FILE,
    SAVE_DIR,
    SAVE_RESULT,
    SAVE_IMAGES,
    UTILS_PATH,
    os.path.join(UTILS_PATH, "database"),
]
base_paths = [os.path.join(BASE_PATH, d) for d in directories]

def initialize_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

initialize_directories(base_paths)
db_path = os.path.join(BASE_PATH, UTILS_PATH, 'database', 'database.db')

db_manager = SQLiteManager(db_path=db_path)

def create_app():
    app = Quart(__name__)

    @app.before_serving
    async def before_serving():
        await db_manager.optimize_sqlite()
        await db_manager.create_table()
        
    @app.before_request
    async def before_request():
        g.db_manager = db_manager
        
    @app.teardown_appcontext
    async def close_db(error):
        await db_manager.close_connection(error)

    app.register_blueprint(blueprint)

    logger.info("Application started successfully")
    return app

