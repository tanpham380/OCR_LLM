import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image


def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lightly increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(gray)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_contrast, (3, 3), 0)

    # Apply adaptive thresholding to handle varying lighting conditions
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Sharpen the image using a kernel
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(adaptive_thresh, -1, kernel)

    # Optionally resize the image to enlarge the QR code
    scale_percent = 175  # Increase by 175%
    width = int(sharpened.shape[1] * scale_percent / 100)
    height = int(sharpened.shape[0] * scale_percent / 100)
    resized = cv2.resize(sharpened, (width, height), interpolation=cv2.INTER_LINEAR)

    return resized

def decode_qr_code(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Save preprocessed image for verification (optional)
    cv2.imwrite("./test.png", preprocessed_image)
    
    # Convert the preprocessed image to PIL Image format for pyzbar to decode
    pil_image = Image.fromarray(preprocessed_image)

    # Decode the QR code
    decoded_objects = decode(pil_image)

    # Print results
    for obj in decoded_objects:
        print("Decoded QR Code Data:", obj.data.decode("utf-8"))
        print("QR Code Location:", obj.polygon)

# Provide the path to the image file
image_path = '/home/gitlab/ocr/static/images/29_09_2024_d67574b7.png'

# Decode the QR code
decode_qr_code(image_path)
