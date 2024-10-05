# import io
# import cv2
# import numpy as np
# from PIL import Image
# from typing import List, Dict, Any

# import torch
# from app_utils.bbox_fix import is_mrz, merge_overlapping_bboxes, remove_duplicate_bboxes
# from app_utils.file_handler import load_and_preprocess_image, save_image
# from app_utils.logging import get_logger
# from app_utils.ocr_package.model.detection.text_detect import TextDect_withRapidocr
# from app_utils.ocr_package.ocr import run_ocr

# from app_utils.ocr_package.model.recognition.processor import (
#     load_processor as load_rec_processor,
# )
# from app_utils.ocr_package.model.recognition.model import load_model as load_rec_model
# from PIL import Image

# from controller.llm_vison_future import VinternOCRModel

# logger = get_logger(__name__)

# import asyncio
# class OcrController:
#     def __init__(self) -> None:
#         self.language_list = ["vi" , "en" ]
#         self.det_processor = TextDect_withRapidocr(text_score = 0.4 , det_use_cuda = True)
#         self.vintern_ocr = VinternOCRModel()

#         self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()

#     async def scan_image(self, image_input: Any, methods: List[str] = [ "package_ocr"] , mat_sau: bool = False ) -> Dict[str, Any]:
#         """
#         Scans the provided image using specified OCR methods.

#         Args:
#             image_input (Any): Path to the image file, NumPy array, or PIL Image.
#             methods (List[str]): List of OCR methods to use.

#         Returns:
#             Dict[str, Any]: OCR results for each method.

#         Raises:
#             Exception: If image loading or preprocessing fails.
#         """
#         processed_img, original_img = load_and_preprocess_image(image_input)
#         if processed_img is None:
#             raise Exception("Failed to load or preprocess the image.")

#         results = {}
#         ocr_methods = {
#             "package_ocr": self._scan_with_package_ocr,
#             # "package_ocr2": self._scan_with_package_ocr2,
#         }

#         for method in methods:
#             if method in ocr_methods:
#                 try:
#                     img_to_use = (
#                         original_img if method == "package_ocr" else processed_img
#                     )
#                     results[method] = await ocr_methods[method](img_to_use , mat_sau)
#                 except Exception as e:
#                     raise Exception(f"Error during scan_image: {e}")
#         return results

     
#     def crop_ocr_box(self, img: np.ndarray, predictions) -> List[Image]:
#         cropped_images = []
        
#         for i, line in enumerate(predictions[0].text_lines):
#             polygon = line.polygon
            
#             # Extract x and y coordinates
#             x_coords = [point[0] for point in polygon]
#             y_coords = [point[1] for point in polygon]
            
#             # Get top-left and bottom-right corners
#             top_left = (min(x_coords), min(y_coords))
#             bottom_right = (max(x_coords), max(y_coords))
            
#             # Cast the coordinates to integers for image slicing
#             top_left = (int(top_left[0]), int(top_left[1]))
#             bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            
#             # Crop the image using numpy slicing
#             cropped_image = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
#             # Convert NumPy array (cropped OpenCV image) to PIL Image
#             pil_image = Image.fromarray(cropped_image)
            
#             # If you want to convert the image (e.g., to grayscale)
#             pil_image = pil_image.convert('L')  # Convert to grayscale (optional)
            
#             cropped_images.append(pil_image)
            
#         return cropped_images
        
#     def draw_packages_ocr_box(self, img: Image.Image, predictions) -> Image.Image:
#         """
#         Draws bounding boxes on the image based on OCR predictions.

#         Args:
#             img (Image.Image): The image to draw on.
#             predictions: OCR predictions containing detected text lines and their bounding boxes.

#         Returns:
#             Image.Image: The image with drawn bounding boxes.
#         """
#         from PIL import ImageDraw

#         draw = ImageDraw.Draw(img)

#         # Define a list of distinct colors
#         colors = [
#             "red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", 
#             "lime", "pink", "teal", "violet"
#         ]

#         # Iterate through text lines and draw bounding boxes
#         num_colors = len(colors)
#         for i, line in enumerate(predictions[0].text_lines):
#             polygon = line.polygon
#             x_coords = [point[0] for point in polygon]
#             y_coords = [point[1] for point in polygon]
#             top_left = (min(x_coords), min(y_coords))
#             bottom_right = (max(x_coords), max(y_coords))

#             # Pick a color from the list in sequence, cycling if necessary
#             color = colors[i % num_colors]

#             # Draw the bounding box with a different color
#             draw.rectangle([top_left, bottom_right], outline=color, width=3)

#         # Convert the image back to BGR for OpenCV compatibility
#         img_resized_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#         return img_resized_bgr

#     async def _scan_with_package_ocr(self, img: np.ndarray, mat_sau: bool = False ) -> str:
#         try:
#             # Convert image to grayscale for OCR processing
#             img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img_original = Image.fromarray(img_gray)
            

#             # Run OCR to get predictions
#             predictions = await asyncio.to_thread(
#                 run_ocr,
#                 [img_original],
#                 [self.language_list],
#                 self.det_processor,
#                 self.rec_model,
#                 self.rec_processor
#             )
#             text_lines = predictions[0].text_lines
#             filtered_text_lines = [line for line in text_lines if line.confidence >= 0.5]
            
#             # text_from_vision_model = await asyncio.to_thread(self.vintern_ocr.process_image, img)
#             # if mat_sau:
#             #     filtered_text_lines = filtered_text_lines[:len(filtered_text_lines)//2]
#             #     filtered_text_lines = [line for line in filtered_text_lines if not is_mrz(line.text)]
#             # bboxes = [list(map(int, line.bbox)) for line in filtered_text_lines]
#             # merged_bboxes =  merge_overlapping_bboxes(bboxes)
#             # cropped_images = []
#             # for bbox in merged_bboxes:
#             #     x_min, y_min, x_max, y_max = bbox
#             #     cropped_img = img[y_min:y_max, x_min:x_max]

#             #     # Convert to RGB if necessary
#             #     if len(cropped_img.shape) == 2 or cropped_img.shape[2] == 1:
#             #         cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
#             #     else:
#             #         cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                
#             #     # Calculate the scaling factor based on the percentage
#             #     scaling_factor = 210 / 100.0
                
#             #     # Get original width and height
#             #     h, w = cropped_img.shape[:2]
                
#             #     # Calculate new dimensions
#             #     new_w = int(w * scaling_factor)
#             #     new_h = int(h * scaling_factor)
                
#             #     # Resize the image based on the scaling percentage
#             #     resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
#             #     # Append the resized image and save it
#             #     cropped_images.append(resized_img)
#             #     save_image(resized_img)
#             # batch_size = 2
#             # text_results = []
#             # for i in range(0, len(cropped_images), batch_size):
#             #     batch_images = cropped_images[i:i+batch_size]
#             #     batch_responses = await asyncio.to_thread(
#             #         self.vintern_ocr.process_images, batch_images
#             #     )
#             #     text_results.extend(batch_responses)

#             # text_from_vision_model = '\n'.join(text_results)
#             # formatted_text = "\n".join(line.text for line in filtered_text_lines)
#             # formatted_section = f"Predicted Text from EssayOCR:\n{formatted_text}\n\n"
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()  
#             formatted_text = ''
#             for line in filtered_text_lines :
#                 print(line.text)
#                 formatted_text = "\n" + self.vintern_ocr.vietname_fix(line.text).get("generated_text" , "")
#                 print(formatted_text)
                
#             # combined_text = formatted_text +"\n\n" + text_from_vision_model
#             # combined_text = self.vintern_ocr.vietname_fix(combined_text)
#             # # Clear GPU memory if available
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()    
#             return formatted_text

#         except Exception as e:
#             raise Exception(f"Error during package OCR: {e}")




#     # async def _scan_with_package_ocr2(self, img: np.ndarray , mat_sau: bool = False) -> str:
#     #     try:
#     #         # Convert image to grayscale for OCR processing
#     #         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #         img_original = Image.fromarray(img_gray)

#     #         # Run OCR to get predictions
#     #         predictions = await asyncio.to_thread(
#     #             run_ocr,
#     #             [img_original],
#     #             [self.language_list],
#     #             self.det_processor,
#     #             self.rec_model,
#     #             self.rec_processor
#     #         )

#     #         # Extract bounding boxes from predictions
#     #         text_lines = predictions[0].text_lines  # Assuming predictions is a list with at least one OCRResult
#     #         filtered_text_lines = text_lines
#     #         if mat_sau:
#     #             filtered_text_lines = [line for line in text_lines if line.confidence >= 0.5]
#     #             filtered_text_lines = filtered_text_lines[:len(filtered_text_lines)//2]
#     #             filtered_text_lines = [line for line in filtered_text_lines if not is_mrz(line.text)]
        
#     #         bboxes = [list(map(int, line.bbox)) for line in filtered_text_lines]  # Convert bbox coordinates to integers

#     #         # Remove duplicate bounding boxes
#     #         # bboxes = remove_duplicate_bboxes(bboxes)

#     #         # Merge overlapping bounding boxes
#     #         merged_bboxes = merge_overlapping_bboxes(bboxes)

#     #         # Now proceed to crop images using the merged bounding boxes
#     #         cropped_images = []
#     #         max_width = 0

#     #         for bbox in merged_bboxes:
#     #             x_min, y_min, x_max, y_max = bbox

#     #             # Crop the image using the bounding box
#     #             cropped_img = img[y_min:y_max, x_min:x_max]

#     #             # Update max_width if this image is wider
#     #             if cropped_img.shape[1] > max_width:
#     #                 max_width = cropped_img.shape[1]

#     #             cropped_images.append(cropped_img)

#     #         # Now, pad images to have the same width
#     #         padded_images = []
#     #         for img_crop in cropped_images:
#     #             height, width = img_crop.shape[:2]
#     #             if width < max_width:
#     #                 # Calculate the amount of padding needed
#     #                 pad_width = max_width - width
#     #                 # Pad the image (pad on the right side)
#     #                 if len(img_crop.shape) == 2:  # Grayscale image
#     #                     img_padded = np.pad(
#     #                         img_crop,
#     #                         ((0, 0), (0, pad_width)),
#     #                         mode='constant',
#     #                         constant_values=255  # White padding
#     #                     )
#     #                 else:  # Color image
#     #                     img_padded = np.pad(
#     #                         img_crop,
#     #                         ((0, 0), (0, pad_width), (0, 0)),
#     #                         mode='constant',
#     #                         constant_values=255  # White padding
#     #                     )
#     #                 padded_images.append(img_padded)
#     #             else:
#     #                 padded_images.append(img_crop)

#     #         # Now stack the padded images vertically
#     #         if padded_images:
#     #             merged_image = np.vstack(padded_images)
#     #         else:
#     #             raise Exception("No text lines found to crop and merge.")

#     #         # Ensure the merged image has a suitable size for OCR
#     #         max_height = 1024  # Adjust based on your OCR model's requirements
#     #         if merged_image.shape[0] > max_height:
#     #             scale_factor = max_height / merged_image.shape[0]
#     #             new_width = int(merged_image.shape[1] * scale_factor)
#     #             merged_image = cv2.resize(merged_image, (new_width, max_height))

#     #         # Convert merged image to RGB if necessary
#     #         if len(merged_image.shape) == 2 or merged_image.shape[2] == 1:
#     #             merged_image = cv2.cvtColor(merged_image, cv2.COLOR_GRAY2RGB)
#     #         else:
#     #             merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)

#     #         # Optionally, save the merged image for debugging
#     #         save_image(merged_image)

#     #         # Send the merged image to the vision LLM OCR
#     #         text_from_vision_model = await asyncio.to_thread(self.vintern_ocr.process_images, [merged_image])

#     #         formatted_text = "\n".join(line.text for line in text_lines)
#     #         formatted_section = f"Predicted Text from EssayOCR:\n{formatted_text}\n\n"
#     #         formatted_section2 = f"Predicted Text from LLM vision:\n{text_from_vision_model}\n\n"
#     #         # Combine texts from both OCR methods
#     #         combined_text = formatted_section + formatted_section2
#     #         # Clear GPU memory if available
#     #         if torch.cuda.is_available():
#     #             torch.cuda.empty_cache()

#     #         print(combined_text)
#     #         return combined_text

#     #     except Exception as e:
#     #         raise Exception(f"Error during package OCR: {e}")
import torch
from transformers import AutoModel, AutoTokenizer , pipeline
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import cv2
from config import DEVICE
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class VinternOCRModel:
    def __init__(self, model_path="/home/gitlab/ocr/Vintern-1B-v3"):
        # """
        # Initialize the model and tokenizer.
        # """
        # self.default_prompt = """trích xuất và trả về văn bản có trong ảnh, không giải thích 
        # Lưu ý: lấy cụ thể các thông tin trên thẻ và trả lời dạng Json \n 
        # - Để " " nếu không có thông tin đó
        # - Chú ý các tường thông tin sau
        #     + Họ và tên \n
        #     + Nơi thường trú (place of residence) \n
        #     + Quê quán ( place of origin , place of birth )\n
        #     + Ngày tháng năm sinh, ngày tháng năm hết hạn và ngày tháng năm kí \n
        # """
        
        # self.model_path = model_path 
        # self.model = AutoModel.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     trust_remote_code=True,
        # ).eval().cuda()
        # self.model = torch.compile(self.model)

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        # self.tokenizer.model_max_length = 4400
        self.corrector = pipeline("text2text-generation", 
                                  model="bmd1905/vietnamese-correction", 
                                  device=DEVICE)
    def vietname_fix(self, text : str) -> str :
        return self.corrector(text, max_length=512)

#     def build_transform(self, input_size):
#         """
#         Create a transformation pipeline to process input images as np.ndarray.
#         """
#         return T.Compose([
#             T.ToTensor(),
#             T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#             T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#         ])

#     def dynamic_preprocess(self, img: np.ndarray, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
#         """
#         Dynamically preprocesses the image (np.ndarray) into blocks based on aspect ratio.
#         """
#         orig_height, orig_width = img.shape[:2]
#         aspect_ratio = orig_width / orig_height

#         # Calculate existing image aspect ratio and find closest
#         target_ratios = [(i, j) for n in range(min_num, max_num + 1)
#                         for i in range(1, n + 1) for j in range(1, n + 1)
#                         if min_num <= i * j <= max_num]
#         target_aspect_ratio = min(target_ratios, key=lambda x: abs(aspect_ratio - (x[0] / x[1])))

#         # Resize the image to fit the aspect ratio
#         target_width = image_size * target_aspect_ratio[0]
#         target_height = image_size * target_aspect_ratio[1]
#         img_resized = cv2.resize(img, (target_width, target_height))

#         # Split the image into blocks
#         block_width = target_width // target_aspect_ratio[0]
#         block_height = target_height // target_aspect_ratio[1]
#         processed_images = [
#             img_resized[y:y + block_height, x:x + block_width]
#             for i in range(target_aspect_ratio[0])
#             for j in range(target_aspect_ratio[1])
#             for x, y in [(i * block_width, j * block_height)]
#         ]

#         if use_thumbnail and len(processed_images) != 1:
#             thumbnail_img = cv2.resize(img, (image_size, image_size))
#             processed_images.append(thumbnail_img)

#         return processed_images

#     def load_image(self, image_file: np.ndarray, input_size=448, max_num=30):
#         """
#         Loads and preprocesses the image (np.ndarray) for OCR.
#         Handles block-based dynamic preprocessing and applies necessary transformations.
#         """
#         transform = self.build_transform(input_size=input_size)

#         try:
#             # Dynamic preprocessing
#             images = self.dynamic_preprocess(image_file, image_size=input_size, use_thumbnail=True, max_num=max_num)

#             # Apply transformations to each block
#             pixel_values = torch.stack([transform(img) for img in images])  # Stack into a batch tensor
#         except Exception as e:
#             raise Exception(f"Error during image transformation: {e}")

#         return pixel_values

#     def process_image(self, image_file: np.ndarray = None, custom_prompt: str = None):
#         """
#         Process the input image (np.ndarray) and generate a response from the model.
#         Optionally, the image can be omitted by passing None, and a custom prompt can be set.
#         """
#         if custom_prompt:
#             prompt = custom_prompt
#         else:
#             prompt = self.default_prompt

#         if image_file is not None:
#             pixel_values = self.load_image(image_file).to(torch.bfloat16).cuda()
#             question = '<image>\n' + prompt
#         else:
#             pixel_values = None  # No image, model may handle this differently
#             question = prompt
#         generation_config = dict(
#             max_new_tokens=2048,
#             do_sample=False,
#             num_beams=1,
#             repetition_penalty=1.2,
#             temperature=0.2,
#             top_p=1.0
#         )

#         response = self.model.chat(
#     self.tokenizer, 
#     pixel_values, 
#     question, 
#     history=None,  # Hoặc history=[]
#     generation_config=generation_config
# )


#         if torch.cuda.is_available():
#             torch.cuda.empty_cache() 
            
            

#         return response





# # class VinternOCRModel:
# #     def __init__(self, model_path="5CD-AI/Vintern-1B-v2"):
# #         # Initialization remains the same
# #         self.default_prompt = "trích xuất và trả về văn bản có trong ảnh, không thêm bất kỳ mô tả hay giải thích nào."

# #         self.model_path = model_path
# #         self.model = AutoModel.from_pretrained(
# #             model_path,
# #             torch_dtype=torch.bfloat16,
# #             low_cpu_mem_usage=True,
# #             trust_remote_code=True,
            
            
            
# #         ).eval().cuda()
# #         self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
# #         self.input_size = 448  # Define input size once

# #     def build_transform(self):
# #         # Build transform without input_size parameter
# #         return T.Compose([
# #             T.ToTensor(),
# #             T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
# #             T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
# #         ])

# #     def load_images(self, image_files: List[np.ndarray]):
# #         """
# #         Loads and preprocesses a list of images for OCR.
# #         """
# #         transform = self.build_transform()
# #         processed_images = []

# #         for img in image_files:
# #             # Apply transformations
# #             img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# #             img_transformed = transform(img_pil)
# #             processed_images.append(img_transformed)

# #         # Stack into a batch tensor
# #         pixel_values = torch.stack(processed_images)
# #         return pixel_values

# #     def process_images(self, image_files: List[np.ndarray], custom_prompt: str = None):
# #         """
# #         Process a batch of input images and generate responses from the model.
# #         """
# #         prompt = custom_prompt if custom_prompt else self.default_prompt
# #         pixel_values = self.load_images(image_files).to(torch.bfloat16).cuda()
# #         batch_size = pixel_values.size(0)

# #         # Create batch of prompts
# #         questions = [f'<image>\n{prompt}'] * batch_size

# #         generation_config = dict(
# #     max_new_tokens=500,
# #     do_sample=False,
# #     num_beams=1,
# #     repetition_penalty=1.2,
# #     temperature=0.2,
# #     top_p=1.0
# # )


# #         # Process the batch
# #         responses = []
# #         with torch.no_grad():
# #             for i in range(batch_size):
# #                 response = self.model.chat(
# #                     self.tokenizer,
# #                     pixel_values[i].unsqueeze(0),
# #                     questions[i],
# #                     generation_config
# #                 )
# #                 responses.append(response)

# #         return responses
