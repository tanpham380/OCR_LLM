# --------
import tempfile
import os
from dotenv import load_dotenv, find_dotenv

from app_utils.util import select_device




# from utils.util import select_device
load_dotenv(find_dotenv())


# --------------------------------

CONF_CONTENT_THRESHOLD = 0.7
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

# FACE_MODEL_PATH = os.path.join(WEIGHTS, "face.pt")

# CORNER_MODEL_PATH = os.path.join(WEIGHTS, "corner.pt")
# CONTENT_MODEL_PATH = os.path.join(WEIGHTS, "content.pt")
CORNER_MODEL_PATH = os.path.join(WEIGHTS, "ID_Card.pt")
CORNER_MODEL_PATH2 = os.path.join(WEIGHTS, "Angle_Deg.pt")

# CORNER_MODEL_PATH2 = os.path.join(WEIGHTS, "CavetDetector_v1.pt")

# CONTENT_MODEL_PATH2 = os.path.join(WEIGHTS, "CCCDFieldsDetector_v1.pt")
# /Users/sentinel/Desktop/testdev/OCR_WITH_LLM/utils/Vocr
# CONFIG_VIETOCR_LOCAL = os.path.join(UTILS_PATH, "Vocr", "config", "vgg-seq2seq.yml")
DEVICE = select_device()
# CONFIG_VIETOCR_MODEL = os.path.join(WEIGHTS, "seq2seqocr.pth")
# MODEL_IMAGE_RECTIFIER= os.path.join(WEIGHTS, "CRDN1000.pkl")

####
os.environ["RECOGNITION_BATCH_SIZE"] = "512"
os.environ["DETECTOR_BATCH_SIZE"] = "36"
os.environ["ORDER_BATCH_SIZE"] = "32"
os.environ["RECOGNITION_STATIC_CACHE"] = "true"
##
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.suppress_errors = True