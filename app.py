from quart import Quart
from app_utils.logging import get_logger
from app_utils.routes import blueprint  # Make sure this matches your actual file/module structure

import atexit
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

def create_app():
    app = Quart(__name__)

    # Initialize directories
    initialize_directories(base_paths)

    # Cleanup function registered to run at exit
    def cleanup():
        logger.info("Cleaning up before shutdown...")

    atexit.register(cleanup)

    # Import and register your blueprint
    app.register_blueprint(blueprint)

    logger.info("Application started successfully")
    return app

