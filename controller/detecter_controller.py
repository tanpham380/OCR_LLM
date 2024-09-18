import math
import os
import re
import cv2
import numpy as np
from typing import Optional, Tuple
from app_utils.logging import get_logger
import config as cfg
from app_utils.image_rectifier_package.image_rectifier import ImageRectify
from controller.ocr_controller import OcrController
from qreader import QReader
from app_utils.file_handler import crop_image_qr, load_and_preprocess_image, process_qr_image, save_image
from deepface import DeepFace


logger = get_logger(__name__)


class Detector:
    UPRIGHT_THRESHOLD = 45  # Threshold to determine upright text
    INVERTED_THRESHOLD = 135  # Threshold to determine inverted text

    def __init__(self) -> None:
        """
        Initializes the Detector class with the necessary components for card detection, OCR, and QR code reading.
        """
        self.card_detecter = ImageRectify()
        self.ocr_controller = OcrController()
        self.qreader = QReader(model_size="l", min_confidence=0.5, reencode_to="cp65001")
        self.ocr_text = None

    
    
    def get_ocr(self) -> OcrController:
        """Returns the OCR controller instance."""
        return self.ocr_controller

    def get_qreader(self) -> QReader:
        """Returns the QR reader instance."""
        return self.qreader

    def detect(self, img_path: str) -> Optional[Tuple[np.ndarray, bool]]:
        """
        Detects and aligns the ID card from the provided image path.

        Args:
            img_path (str): Path to the image file.

        Returns:
            np.ndarray: Aligned image.

        Raises:
            Exception: If the image loading or detection fails.
        """
        try :
            img , _ = load_and_preprocess_image(img_path)
            
            detections , is_front = self.card_detecter.detect(img)
            return detections , is_front
        except Exception as e:
            raise e


    def read_QRcode(self, image: np.ndarray) -> str:
        """
        Reads and decodes the QR code from the provided image.

        Args:
            image (np.ndarray): Image array containing the QR code.

        Returns:
            str: Decoded QR code text.

        Raises:
            Exception: If QR code reading fails.
        """
        try:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            qr_detections = self.qreader.detect(image)
            if not qr_detections:
                raise Exception("No QR code detected in the image.")
            text_qr = self.qreader.decode(image, qr_detections[0])
            if text_qr is None:
                cropped_qr = crop_image_qr(image,qr_detections[0]  )
                text_qr = self.qreader.detect_and_decode(cropped_qr)
                if text_qr is None or text_qr == (None,) or text_qr == ():
                    new_img = process_qr_image(cropped_qr)
                    text_qr = self.qreader.detect_and_decode(new_img)
                    if text_qr is None or text_qr == (None,) or text_qr == ():
                        gray_image = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                        text_qr = self.qreader.detect_and_decode(gray_image)
            if text_qr is None or text_qr == (None,) or text_qr == ():
                text_qr = ""
            return text_qr
        except Exception as ex:
            logger.info(ex)
            return ""


    def detect_face_orientation(self, image: np.ndarray) -> Optional[bool]:
        """
        Detects the face orientation to determine if the card is upright or upside down.

        Args:
            image (np.ndarray): Image array to detect face orientation.

        Returns:
            Optional[bool]: True if the face is upright, False if upside down, or None if no face is detected.
        """
        try:
            faces = DeepFace.extract_faces(image, enforce_detection=False, detector_backend='yolov8', anti_spoofing = False)

            if not faces:
                raise Exception(f"No facial area found for the detected face.")
            facial_area = faces[0].get("facial_area")
            if not facial_area:
                raise Exception(f"No facial area found for the detected face.")

            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
            face_center_y = y + h / 2
            img_height = image.shape[0]
            is_upright = face_center_y < img_height / 2
            
            return is_upright

        except Exception as ex:
            raise Exception(f"An error occurred during face detection: {str(ex)}")


    def detect_card_orientation(self, image: np.ndarray) -> Optional[bool]:
        """
        Detects if the card image is upside down or upright using OCR and MRZ detection.

        Args:
            image (np.ndarray): Image array for OCR.

        Returns:
            Optional[bool]: True if the image is upside down, False if upright, or None if OCR fails.
        """
        try:
            if image is None:
                raise ValueError("Error: No picture ID card provided.")
            self.ocr_text = self.ocr_controller.reader.readtext(image)
            if not self.ocr_text:
                raise ValueError("OCR text is not available.")
            mrz_pattern = r'<<'
            found_mrz = False
            for detection in self.ocr_text:
                box, text, prob = detection
                if re.search(mrz_pattern, text):
                    _, y_bottom = box[2]  
                    if y_bottom > image.shape[0] // 2:
                        found_mrz = True
                        break
            if found_mrz:
                return False 
            else:
                return True 

        except Exception as ex:
            print(f"An error occurred during card orientation detection: {str(ex)}")
            return None
