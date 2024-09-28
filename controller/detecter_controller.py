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
from app_utils.file_handler import (
    crop_image_qr,
    load_and_preprocess_image,
    process_qr_image,
    save_image,
    Enchane_Qr_image
)

# from deepface import DeepFace


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
        self.qreader = QReader(
            model_size="l", min_confidence=0.5, reencode_to="cp65001"
        )

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
        try:
            img, _ = load_and_preprocess_image(img_path)

            detections, is_front = self.card_detecter.detect(img)
            return detections, is_front
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
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            qr_detections = self.qreader.detect(image)

            if not qr_detections:
                raise Exception("No QR code detected in the image.")
            text_qr = self.qreader.decode(image, qr_detections[0])

            if text_qr is None:
                cropped_qr = crop_image_qr(image, qr_detections[0])
                new_img = process_qr_image(cropped_qr)
                text_qr = self.qreader.detect_and_decode(new_img)
                if text_qr is None or text_qr == (None,) or text_qr == ():
                    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                    text_qr = self.qreader.detect_and_decode(gray)
                    if text_qr is None or text_qr == (None,) or text_qr == ():
                        gray_image = Enchane_Qr_image(new_img)
                        text_qr = self.qreader.detect_and_decode(gray_image)
            if text_qr is None or text_qr == (None,) or text_qr == ():
                text_qr = ""
            return text_qr
        except Exception as ex:
            logger.info(ex)
            return ""

