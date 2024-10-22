# import torch

# from transformers import (
#     AutoModel,
#     AutoTokenizer,

# )
# import torchvision.transforms as T
# from torchvision.transforms.functional import InterpolationMode
# import numpy as np
# import cv2
# from PIL import Image
# from torchvision.transforms.functional import InterpolationMode
# import torch.nn as nn

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)

# class VinternOCRModel:
#     def __init__(self, model_path="app_utils/weights/Vintern-4B-v1", device=None):
#         # ...
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_path = model_path

#         # Load the tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_path, trust_remote_code=True, use_fast=False
#         )
#         self.tokenizer2 = AutoTokenizer.from_pretrained(
#             model_path, trust_remote_code=True, use_fast=False
#         )
#         self.default_prompt = (
#             "Hãy trích xuất toàn bộ thông tin từ bức ảnh này theo đúng thứ tự và nội dung như trong ảnh, đảm bảo đầy đủ và chính xác. "
#             "Không thêm bất kỳ bình luận nào khác. "
#             "Lưu ý: Đối với 'Nơi thường trú' và 'Quê quán', hãy trích xuất đầy đủ địa chỉ như trong ảnh, bao gồm cả xã, huyện, tỉnh.\n"
#         )

#         # Load the model
#         self.model = AutoModel.from_pretrained(
#             model_path,
#             torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#             # device_map = "auto"
#         ).to(self.device)
#         self.tokenizer.model_max_length = 8196

        

#     def build_transform(self, input_size):
#         """
#         Create a transformation pipeline to process input images as PIL Images.
#         """
#         MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
#         transform = T.Compose([
#             T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#             T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#             T.ToTensor(),
#             T.Normalize(mean=MEAN, std=STD)
#         ])
#         return transform

#     def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
#         """
#         Find the closest aspect ratio from target ratios.
#         """
#         best_ratio_diff = float('inf')
#         best_ratio = (1, 1)
#         area = width * height
#         for ratio in target_ratios:
#             target_aspect_ratio = ratio[0] / ratio[1]
#             ratio_diff = abs(aspect_ratio - target_aspect_ratio)
#             if ratio_diff < best_ratio_diff:
#                 best_ratio_diff = ratio_diff
#                 best_ratio = ratio
#             elif ratio_diff == best_ratio_diff:
#                 if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
#                     best_ratio = ratio
#         return best_ratio

#     def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
#         """
#         Dynamically preprocess the image into blocks based on aspect ratio.
#         """
#         orig_width, orig_height = image.size
#         aspect_ratio = orig_width / orig_height

#         # Calculate possible grid sizes
#         target_ratios = set(
#             (i, j) for n in range(min_num, max_num + 1)
#             for i in range(1, n + 1)
#             for j in range(1, n + 1)
#             if i * j <= max_num and i * j >= min_num
#         )
#         target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

#         # Find the closest aspect ratio to the target
#         target_aspect_ratio = self.find_closest_aspect_ratio(
#             aspect_ratio, target_ratios, orig_width, orig_height, image_size)

#         # Calculate the target width and height
#         target_width = image_size * target_aspect_ratio[0]
#         target_height = image_size * target_aspect_ratio[1]
#         blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

#         # Resize the image
#         resized_img = image.resize((target_width, target_height))

#         # Split the image into blocks
#         processed_images = []
#         for i in range(blocks):
#             box = (
#                 (i % target_aspect_ratio[0]) * image_size,
#                 (i // target_aspect_ratio[0]) * image_size,
#                 ((i % target_aspect_ratio[0]) + 1) * image_size,
#                 ((i // target_aspect_ratio[0]) + 1) * image_size
#             )
#             split_img = resized_img.crop(box)
#             processed_images.append(split_img)

#         if use_thumbnail and len(processed_images) != 1:
#             thumbnail_img = image.resize((image_size, image_size))
#             processed_images.append(thumbnail_img)

#         return processed_images
#     def load_image(self, image_file, input_size=448, max_num=6):
#         """
#         Loads and preprocesses the image for OCR.
#         Handles block-based dynamic preprocessing and applies necessary transformations.
#         """
#         if isinstance(image_file, np.ndarray):
#             # Convert NumPy array to PIL Image
#             image = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))
#         elif isinstance(image_file, Image.Image):
#             image = image_file
#         elif isinstance(image_file, torch.Tensor):
#             # Convert PyTorch tensor to PIL Image
#             image = Image.fromarray(image_file.cpu().numpy().astype(np.uint8))
#         else:
#             raise ValueError("Unsupported image format. Provide a NumPy array, PIL Image, or PyTorch tensor.")

#         transform = self.build_transform(input_size=input_size)

#         # Dynamic preprocessing
#         images = self.dynamic_preprocess(
#             image, image_size=input_size, use_thumbnail=True, max_num=max_num
#         )

#         # Apply transformations to each block
#         try:
#             pixel_values = [transform(img) for img in images]
#             pixel_values = torch.stack(pixel_values)
#         except Exception as e:
#             raise Exception(f"Error during image transformation: {e}")

#         return pixel_values
#     def process_image(self, image_file=None, custom_prompt=None, input_size=448, max_num=12):
#         """
#         Process the input image and generate a response from the model.
#         Optionally, the image can be omitted by passing None, and a custom prompt can be set.
#         """
#         prompt = custom_prompt if custom_prompt else self.default_prompt

#         if image_file is not None:
#             pixel_values = self.load_image(image_file, input_size, max_num)
#             pixel_values = pixel_values.to(torch.bfloat16).to(self.device) 
#             question = "<image>\n" + prompt
#         else:
#             pixel_values = None  # No image provided
#             question = prompt

#         generation_config = {
#             "max_new_tokens": 8196,
#             "do_sample": False,
#             "num_beams": 2,
#             "repetition_penalty": 2.0,            
#         }

#         response = self.model.chat(
#             self.tokenizer,
#             pixel_values,
#             question,
#             history=None,
#             generation_config=generation_config,
#         )

#         # Clear GPU cache after processing
#         torch.cuda.empty_cache()

#         return response


import torch

from transformers import (
    AutoModel,
    AutoTokenizer,

)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class VinternOCRModel:
    def __init__(self, model_path="app_utils/weights/Vintern-3B-v1-phase4-ocr", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

        self.default_prompt = (
            "Hãy trích xuất toàn bộ thông tin từ bức ảnh này theo đúng thứ tự và nội dung như trong ảnh, đảm bảo đầy đủ và chính xác. "
            "Không thêm bất kỳ bình luận nào khác. "
            "Lưu ý: Đối với 'Nơi thường trú' và 'Quê quán', hãy trích xuất đầy đủ địa chỉ như trong ảnh, bao gồm cả xã, huyện, tỉnh.\n"
        )

        # Load the model
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map = "auto"
        )#.to(self.device)
        self.tokenizer.model_max_length = 65728

        # Áp dụng DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def build_transform(self, input_size):
        """
        Create a transformation pipeline to process input images as PIL Images.
        """
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """
        Find the closest aspect ratio from target ratios.
        """
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """
        Dynamically preprocess the image into blocks based on aspect ratio.
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate possible grid sizes
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height))

        # Split the image into blocks
        processed_images = []
        for i in range(blocks):
            box = (
                (i % target_aspect_ratio[0]) * image_size,
                (i // target_aspect_ratio[0]) * image_size,
                ((i % target_aspect_ratio[0]) + 1) * image_size,
                ((i // target_aspect_ratio[0]) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images
    def load_image(self, image_file, input_size=448, max_num=6):
        """
        Loads and preprocesses the image for OCR.
        Handles block-based dynamic preprocessing and applies necessary transformations.
        """
        if isinstance(image_file, np.ndarray):
            # Convert NumPy array to PIL Image
            image = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))
        elif isinstance(image_file, Image.Image):
            image = image_file
        elif isinstance(image_file, torch.Tensor):
            # Convert PyTorch tensor to PIL Image
            image = Image.fromarray(image_file.cpu().numpy().astype(np.uint8))
        else:
            raise ValueError("Unsupported image format. Provide a NumPy array, PIL Image, or PyTorch tensor.")

        transform = self.build_transform(input_size=input_size)

        # Dynamic preprocessing
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )

        # Apply transformations to each block
        try:
            pixel_values = [transform(img) for img in images]
            pixel_values = torch.stack(pixel_values)
        except Exception as e:
            raise Exception(f"Error during image transformation: {e}")

        return pixel_values
    def process_image(self, image_file=None, custom_prompt=None, input_size=448, max_num=30):
        """
        Process the input image and generate a response from the model.
        Optionally, the image can be omitted by passing None, and a custom prompt can be set.
        """
        prompt = custom_prompt if custom_prompt else self.default_prompt

        if image_file is not None:
            pixel_values = self.load_image(image_file, input_size, max_num)
            pixel_values = pixel_values.to(torch.bfloat16)#.to(self.device) 
            question = "<image>\n" + prompt
        else:
            pixel_values = None  # No image provided
            question = prompt

        generation_config = {
            "max_new_tokens": 8196,
            "do_sample": False,
            "num_beams": 2,
            "repetition_penalty": 2.0,            
        }

        response = self.model.module.chat(
    self.tokenizer,
    pixel_values,
    question,
    history=None,
    generation_config=generation_config,
)

        # Clear GPU cache after processing
        torch.cuda.empty_cache()

        return response
    
    
    
    
    
    def process_images(self, image_files=None, custom_prompt=None, input_size=448, max_num=6):
        """
        Processes a list of input images and generates a response from the model.
        Optionally, the image list can be omitted by passing None, and a custom prompt can be set.
        """
        prompt = custom_prompt if custom_prompt else self.default_prompt

        if image_files is not None:
            all_pixel_values = []
            for image_file in image_files:
                pixel_values = self.load_image(image_file, input_size, max_num)
                pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
                all_pixel_values.append(pixel_values)
            
            # Concatenate pixel values along the batch dimension
            pixel_values = torch.cat(all_pixel_values, dim=0)
            question = "<image>\n" + prompt
        else:
            pixel_values = None  # No images provided
            question = prompt

        generation_config = {
            "max_new_tokens": 8196,
            "do_sample": False,
            "num_beams": 2,
            "repetition_penalty": 2.0,            
        }

        response = self.model.module.chat(
            self.tokenizer,
            pixel_values,
            question,
            history=None,
            generation_config=generation_config,
        )

        # Clear GPU cache after processing
        torch.cuda.empty_cache()

        return response