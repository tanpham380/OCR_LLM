import cv2
import numpy as np
from PIL import Image, ImageEnhance
import easyocr
from typing import List, Dict, Any, Optional
from app_utils.file_handler import load_and_preprocess_image, save_image
from app_utils.logging import get_logger
from app_utils.ocr_package.ocr import run_ocr
from app_utils.ocr_package.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from app_utils.ocr_package.model.recognition.processor import load_processor as load_rec_processor
from app_utils.ocr_package.model.recognition.model import load_model as load_rec_model

logger = get_logger(__name__)

class OcrController:
    def __init__(self) -> None:
        self.language_list = ["vi", "en"]
        self.reader = easyocr.Reader(self.language_list, gpu=True)
        self.det_processor, self.det_model = load_det_processor(), load_det_model()
        self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
    
    async def scan_image(self, image_input: Any, methods: List[str] = ["easyocr", "package_ocr"]) -> Dict[str, Any]:
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
            "easyocr": self._scan_with_easyocr,
            "package_ocr": self._scan_with_package_ocr
        }

        for method in methods:
            if method in ocr_methods:
                try:
                    img_to_use = original_img if method == "package_ocr" else processed_img
                    results[method] = await ocr_methods[method](img_to_use)
                except Exception as e:
                    raise Exception(f"Error during scan_image: {e}") 
        return results


    async def _scan_with_easyocr(self, img: np.ndarray) -> str:
        """
        Performs OCR on the image using EasyOCR.

        Args:
            img (np.ndarray): Image array.

        Returns:
            str: Extracted text.

        Raises:
            Exception: If OCR fails.
        """
        try:
            results = self.reader.readtext(
                img,
                detail=1,
                contrast_ths=0.7,
                adjust_contrast=0.5,
                text_threshold=0.6,
                link_threshold=0.4,
                decoder="wordbeamsearch",
                paragraph=False,
            )
            
            if not results:
                return ""
            
            ocr_str = ' '.join(result[1] for result in results)
            return ocr_str
        except Exception as e:
            raise Exception(f"Error during EasyOCR: {e}")

    # async def _scan_with_package_ocr(self, img: np.ndarray) -> str:
    #     """
    #     Performs OCR on the image using package OCR methods.

    #     Args:
    #         img (np.ndarray): Image array.

    #     Returns:
    #         str: Extracted text.

    #     Raises:
    #         Exception: If OCR or image processing fails.
    #     """
    #     try:
    #         img_original = Image.fromarray(img)
    #         desired_size = (2048, 1024)
    #         margin = 100

    #         if img_original.width > (desired_size[0] - margin * 2) or img_original.height > (desired_size[1] - margin * 2):
    #             aspect_ratio = min((desired_size[0] - margin * 2) / img_original.width, (desired_size[1] - margin * 2) / img_original.height)
    #             img_original = img_original.resize((int(img_original.width * aspect_ratio), int(img_original.height * aspect_ratio)), Image.Resampling.LANCZOS)

    #         new_img = Image.new("RGB", desired_size, (255, 255, 255))
    #         new_img.paste(img_original, ((desired_size[0] - img_original.width) // 2, (desired_size[1] - img_original.height) // 2))
    #         save_image(new_img)
    #         predictions = run_ocr([new_img], [self.language_list], self.det_model, self.det_processor, self.rec_model, self.rec_processor)

    #         formatted_text = "\n".join(line.text for line in predictions[0].text_lines)
    #         return formatted_text
    #     except Exception as e:
    #         logger.error(f"Error during package OCR: {e}")
    #         raise Exception(f"Error during package OCR: {e}")
    # async def _scan_with_package_ocr(self, img: np.ndarray) -> str:
    #     """
    #     Performs OCR on the image using package OCR methods.

    #     Args:
    #         img (np.ndarray): Image array.

    #     Returns:
    #         str: Extracted text.

    #     Raises:
    #         Exception: If OCR or image processing fails.
    #     """
    #     try:
    #         img_original = Image.fromarray(img)  # Convert the image array to a Pillow image
    #         desired_size = (1024, 1024) 
    #         aspect_ratio = min(desired_size[0] / img_original.width, desired_size[1] / img_original.height)
    #         new_width = int(img_original.width * aspect_ratio)
    #         new_height = int(img_original.height * aspect_ratio)
    #         img_resized = img_original.resize((new_width, new_height), Image.Resampling.LANCZOS)
    #         new_img = Image.new("RGB", desired_size, (255, 255, 255))
    #         new_img.paste(img_resized, ((desired_size[0] - new_width) // 2, (desired_size[1] - new_height) // 2))
    #         save_image(new_img)
    #         predictions = run_ocr([new_img], [self.language_list], self.det_model, self.det_processor, self.rec_model, self.rec_processor)
    #         formatted_text = "\n".join(line.text for line in predictions[0].text_lines)
    #         return formatted_text

    #     except Exception as e:
    #         logger.error(f"Error during package OCR: {e}")
    #         raise Exception(f"Error during package OCR: {e}")
            

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
            img_original = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            desired_size = (1024, 1024)
            aspect_ratio = min(desired_size[0] / img_original.width, desired_size[1] / img_original.height)
            new_width = int(img_original.width * aspect_ratio)
            new_height = int(img_original.height * aspect_ratio)
            img_resized = img_original.resize((new_width, new_height), Image.Resampling.LANCZOS)
            new_img = Image.new("RGB", desired_size, (255, 255, 255))

            # Calculate the position to paste the resized image onto the white background
            paste_position = ((desired_size[0] - new_width) // 2, (desired_size[1] - new_height) // 2)

            # If img_resized is not in RGB mode, convert it to RGB
            if img_resized.mode != 'RGB':
                img_resized = img_resized.convert('RGB')

            # Paste the resized image onto the white background
            new_img.paste(img_resized, paste_position)
            predictions = run_ocr(
                [new_img],
                [self.language_list],
                self.det_model,
                self.det_processor,
                self.rec_model,
                self.rec_processor
            )

            # Extract and format text
            formatted_text = "\n".join(line.text for line in predictions[0].text_lines)
            return formatted_text

        except Exception as e:
            raise Exception(f"Error during package OCR: {e}")

    def draw_packages_ocr_box(img: np.ndarray, predictions, color=(255, 0, 0)) -> np.ndarray:
        """
        Draws bounding boxes on the image based on OCR predictions.

        Args:
            img (np.ndarray): Image array.
            predictions (Any): OCR predictions containing bounding boxes.
            color (tuple): Color of the bounding box.

        Returns:
            np.ndarray: Image with bounding boxes drawn.
        """
        from PIL import ImageDraw
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        if isinstance(predictions, list):
            for pred in predictions:
                if hasattr(pred, 'bboxes'):
                    for bbox in pred.bboxes:
                        draw.rectangle(bbox, outline=color, width=2)
                elif hasattr(pred, 'bbox'):
                    draw.rectangle(pred.bbox, outline=color, width=2)
                elif hasattr(pred, 'polygon'):
                    draw.polygon(pred.polygon, outline=color, width=2)
        elif hasattr(predictions, 'bboxes'):
            for bbox in predictions.bboxes:
                draw.rectangle(bbox, outline=color, width=2)

        return np.array(pil_img)
