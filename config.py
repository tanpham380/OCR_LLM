# --------
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
import torch

from app_utils.util import select_device




# from utils.util import select_device
load_dotenv(find_dotenv())


# --------------------------------

CONF_CONTENT_THRESHOLD = 0.75
IOU_CONTENT_THRESHOLD = 0.7
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# CONF_CORNER_THRESHOLD = 0.8
# IOU_CORNER_THRESHOLD = 0.5
# --------------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# os.path.join(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH_FILE = os.path.join(BASE_PATH, "logs")
SAVE_DIR = os.path.join(BASE_PATH, "static")
SAVE_RESULT = os.path.join(SAVE_DIR, "result")
SAVE_IMAGES = os.path.join(SAVE_DIR, "images")
UTILS_PATH = os.path.join(BASE_PATH, "app_utils")
DB_PATH = os.path.join(UTILS_PATH ,"database" , "db.sqlite3")
TEMP_DIR = tempfile.gettempdir()

API_KEY = os.getenv("API_KEY")
URL_API_LLM = os.getenv("URL_API_LLM")

# --------
WEIGHTS = os.path.join(UTILS_PATH, "weights")

CORNER_MODEL_PATH = os.path.join(WEIGHTS, "ID_Card.pt")

DEVICE = select_device()

####
os.environ["RECOGNITION_BATCH_SIZE"] = "512"
os.environ["DETECTOR_BATCH_SIZE"] = "36"
os.environ["ORDER_BATCH_SIZE"] = "32"
