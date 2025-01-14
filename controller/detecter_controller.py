import math
import os
import re
import cv2
import numpy as np
from typing import Any, Optional, Tuple
from app_utils.logging import get_logger
import config as cfg
from app_utils.image_rectifier_package.image_rectifier import ImageRectify
from qreader import QReader
from app_utils.file_handler import (
    crop_image_qr,
    load_and_preprocess_image,
    process_qr_image,
    save_image,
    scale_up_img
)



logger = get_logger(__name__)


class Detector:
    UPRIGHT_THRESHOLD = 45  # Threshold to determine upright text
    INVERTED_THRESHOLD = 135  # Threshold to determine inverted text

    def __init__(self) -> None:
        """
        Initializes the Detector class with improved QR reading for Vietnamese text
        """
        self.card_detecter = ImageRectify(crop_expansion_factor = 0.0)
        
        # Updated QReader config for Vietnamese support
        self.qreader = QReader(
            model_size="l",  # Use large model for better accuracy
            min_confidence=0.5,  # Lower threshold for better detection
    reencode_to=['big5', 'shift-jis', 'cp65001' ]  # Vietnamese text encoding
        )
        
        self.ocr_text = None

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
            if img is None:
                raise Exception("Failed to load or preprocess the image.")
            detections, is_front = self.card_detecter.detect(img)
            if detections is None:
                return img
            return detections, is_front
        except Exception as e:
            raise e
            

    def read_QRcode(self, image: np.ndarray) -> str:
        """
        Reads and decodes the QR code from the provided image.

        Args:
            image (np.ndarray): Image array containing the QR code.

        Returns:
            str: Decoded QR code text or an empty string if reading fails.
        """

        def process_and_decode(img: np.ndarray, detect: Any = None) -> str:
            detections = detect if detect is not None else self.qreader.detect(img)
            if detections:
                return self.qreader.decode(img, detections[0])
            logger.info("No QR code detected in the image.")
            return ""

        try:
            text_qr = process_and_decode(image)
            if text_qr:
                return text_qr

            detections = self.qreader.detect(image)
            
            if detections:
                cropped_qr = crop_image_qr(image, detections[0])
                enhanced_img = process_qr_image(cropped_qr)

                text_qr = process_and_decode(enhanced_img, detections)
                if text_qr:
                    return text_qr

                scaled_img = scale_up_img(enhanced_img, 480)
                return process_and_decode(scaled_img) or ""
            return "" 
        except Exception as ex:
            logger.error(f"QR Code reading failed: {ex}")
            return ""
