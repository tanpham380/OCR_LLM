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
        Initializes the Detector class with the necessary components for card detection, OCR, and QR code reading.
        """
        self.card_detecter = ImageRectify(crop_expansion_factor = 0.0)
        self.qreader = QReader(
            model_size="l", min_confidence=0.7, reencode_to="cp65001"
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
            

    # def read_QRcode(self, image: np.ndarray) -> str:
    #     """
    #     Reads and decodes the QR code from the provided image.

    #     Args:
    #         image (np.ndarray): Image array containing the QR code.

    #     Returns:
    #         str: Decoded QR code text.

    #     Raises:
    #         Exception: If QR code reading fails.
    #     """
    #     try:
    #         if len(image.shape) == 2:
    #             image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #         elif len(image.shape) == 3 and image.shape[2] == 4:
    #             image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    #         qr_detections = self.qreader.detect(image)

    #         if not qr_detections:
    #             raise Exception("No QR code detected in the image.")
    #         text_qr = self.qreader.decode(image, qr_detections[0])

    #         if text_qr is None:
    #             cropped_qr = crop_image_qr(image, qr_detections[0])
                
    #             new_img = process_qr_image(cropped_qr)
                
    #             qr_detections = self.qreader.detect(new_img)
    #             text_qr = self.qreader.decode(new_img, qr_detections[0])
    #             # text_qr = self.qreader.detect_and_decode(new_img)
    #             if text_qr is None or text_qr == (None,) or text_qr == ():
    #                 gray_image = scale_up_img(new_img, 480)
    #                 qr_detections = self.qreader.detect(gray_image)
    #                 text_qr = self.qreader.decode(new_img, qr_detections[0])
    #         if text_qr is None or text_qr == (None,) or text_qr == ():
    #             text_qr = ""
    #         return text_qr
    #     except Exception as ex:
    #         logger.info(ex)
    #         return ""



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
