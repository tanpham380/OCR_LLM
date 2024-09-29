import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
# import easyocr
from typing import List, Dict, Any, Optional
from app_utils.file_handler import load_and_preprocess_image, save_image
from app_utils.logging import get_logger
from app_utils.ocr_package.model.detection.text_detect import TextDect_withRapidocr
from app_utils.ocr_package.ocr import run_ocr
# from app_utils.ocr_package.model.detection.model import (
#     load_model as load_det_model,
#     load_processor as load_det_processor,
# )
from app_utils.ocr_package.model.recognition.processor import (
    load_processor as load_rec_processor,
)
from app_utils.ocr_package.model.recognition.model import load_model as load_rec_model

logger = get_logger(__name__)


class OcrController:
    def __init__(self) -> None:
        self.language_list = ["vi"  , "en"] #, "en"
        # self.reader = easyocr.Reader(self.language_list, gpu=True)
        # self.det_processor, self.det_model = load_det_processor(), load_det_model()
        self.det_processor = TextDect_withRapidocr(text_score = 0.4 , det_use_cuda = False)
        self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
        # self.rec_model.decoder.model = torch.compile(self.rec_model.decoder.model)

    async def scan_image(
        self, image_input: Any, methods: List[str] = ["easyocr", "package_ocr"]
    ) -> Dict[str, Any]:
        """
        Scans the provided image using specified OCR methods.

        Args:
            image_input (Any): Path to the image file, NumPy array, or PIL Image.
            methods (List[str]): List of OCR methods to use.

        Returns:
            Dict[str, Any]: OCR results for each method.

        Raises:
            Exception: If image loading or preprocessing fails.
        """
        processed_img, original_img = load_and_preprocess_image(image_input)
        if processed_img is None:
            raise Exception("Failed to load or preprocess the image.")

        results = {}
        ocr_methods = {
            # "easyocr": self._scan_with_easyocr,
            "package_ocr": self._scan_with_package_ocr,
        }

        for method in methods:
            if method in ocr_methods:
                try:
                    img_to_use = (
                        original_img if method == "package_ocr" else processed_img
                    )
                    results[method] = await ocr_methods[method](img_to_use)
                except Exception as e:
                    raise Exception(f"Error during scan_image: {e}")
        return results

    # async def _scan_with_easyocr(self, img: np.ndarray) -> str:
    #     """
    #     Performs OCR on the image using EasyOCR.

    #     Args:
    #         img (np.ndarray): Image array.

    #     Returns:
    #         str: Extracted text.

    #     Raises:
    #         Exception: If OCR fails.
    #     """
    #     try:
    #         results = self.reader.readtext(
    #             img,
    #             detail=1,
    #             contrast_ths=0.7,
    #             adjust_contrast=0.5,
    #             text_threshold=0.6,
    #             link_threshold=0.4,
    #             decoder="wordbeamsearch",
    #             paragraph=False,
    #         )

    #         if not results:
    #             return ""

    #         ocr_str = " ".join(result[1] for result in results)
    #         return ocr_str
    #     except Exception as e:
    #         raise Exception(f"Error during EasyOCR: {e}")

    async def _scan_with_package_ocr(self, img: np.ndarray) -> str:
        """
        Performs OCR on the image using package OCR methods.

        Args:
            img (np.ndarray): Image array.

        Returns:
            str: Extracted text.

        Raises:
            Exception: If OCR or image processing fails.
        """
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # _, img_black_white = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
             
            img_original = Image.fromarray(img_gray)
            # img_original = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) , mode="L")
            # img_resized = img_original.resize((1000, 1000), Image.Resampling.LANCZOS)
            # background_size = (1280, 1080)
            # new_img = Image.new("RGB", background_size, (255, 255, 255))
            # paste_position = (
            #     (background_size[0] - img_resized.width) // 2,
            #     (background_size[1] - img_resized.height) // 2,
            # )
            # new_img.paste(img_resized, paste_position)
            predictions = run_ocr(
                [img_original],
                [self.language_list],
                self.det_processor,
                self.rec_model,
                self.rec_processor,
            )
            image_with_boxes = self.draw_packages_ocr_box(img_original, predictions)
            save_image(image_with_boxes)
            formatted_text = "\n".join(line.text for line in predictions[0].text_lines)            
            return formatted_text

        except Exception as e:
            raise Exception(f"Error during package OCR: {e}")


    def draw_packages_ocr_box(self, img: Image.Image, predictions) -> Image.Image:
        """
        Draws bounding boxes on the image based on OCR predictions.

        Args:
            img (Image.Image): The image to draw on.
            predictions: OCR predictions containing detected text lines and their bounding boxes.

        Returns:
            Image.Image: The image with drawn bounding boxes.
        """
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)

        # Define a list of distinct colors
        colors = [
            "red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", 
            "lime", "pink", "teal", "violet"
        ]

        # Iterate through text lines and draw bounding boxes
        num_colors = len(colors)
        for i, line in enumerate(predictions[0].text_lines):
            polygon = line.polygon
            x_coords = [point[0] for point in polygon]
            y_coords = [point[1] for point in polygon]
            top_left = (min(x_coords), min(y_coords))
            bottom_right = (max(x_coords), max(y_coords))

            # Pick a color from the list in sequence, cycling if necessary
            color = colors[i % num_colors]

            # Draw the bounding box with a different color
            draw.rectangle([top_left, bottom_right], outline=color, width=3)

        # Convert the image back to BGR for OpenCV compatibility
        img_resized_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        return img_resized_bgr



