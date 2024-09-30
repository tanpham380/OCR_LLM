from typing import Any, Dict
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import cv2
import io

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class VinternOCRModel:
    def __init__(self, model_path="5CD-AI/Vintern-1B-v2"):
        """
        Initialize the model and tokenizer.
        """
        self.default_prompt = "Trích thông tin từ ảnh, không giải thích"
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def build_transform(self, input_size):
        """
        Create a transformation pipeline to process input images as np.ndarray.
        """
        return T.Compose([
            T.ToTensor(),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def dynamic_preprocess(self, img: np.ndarray, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """
        Dynamically preprocesses the image (np.ndarray) into blocks based on aspect ratio.
        """
        orig_height, orig_width = img.shape[:2]
        aspect_ratio = orig_width / orig_height

        # Calculate existing image aspect ratio and find closest
        target_ratios = [(i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if min_num <= i * j <= max_num]
        target_aspect_ratio = min(target_ratios, key=lambda x: abs(aspect_ratio - (x[0] / x[1])))

        # Resize the image to fit the aspect ratio
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        img_resized = cv2.resize(img, (target_width, target_height))

        # Split the image into blocks
        block_width = target_width // target_aspect_ratio[0]
        block_height = target_height // target_aspect_ratio[1]
        processed_images = [
            img_resized[y:y + block_height, x:x + block_width]
            for i in range(target_aspect_ratio[0])
            for j in range(target_aspect_ratio[1])
            for x, y in [(i * block_width, j * block_height)]
        ]

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = cv2.resize(img, (image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def load_image(self, image_file: np.ndarray, input_size=448, max_num=12):
        """
        Loads and preprocesses the image (np.ndarray) for OCR.
        Handles block-based dynamic preprocessing and applies necessary transformations.
        """
        transform = self.build_transform(input_size=input_size)

        try:
            # Dynamic preprocessing
            images = self.dynamic_preprocess(image_file, image_size=input_size, use_thumbnail=True, max_num=max_num)

            # Apply transformations to each block
            pixel_values = torch.stack([transform(img) for img in images])  # Stack into a batch tensor
        except Exception as e:
            raise Exception(f"Error during image transformation: {e}")

        return pixel_values

    def process_image(self, image_file: np.ndarray = None, custom_prompt: str = None):
        """
        Process the input image (np.ndarray) and generate a response from the model.
        Optionally, the image can be omitted by passing None, and a custom prompt can be set.
        """
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self.default_prompt

        # If image_file is provided, process it, otherwise skip image handling
        if image_file is not None:
            pixel_values = self.load_image(image_file).to(torch.bfloat16).cuda()
            question = '<image>\n' + prompt
        else:
            pixel_values = None  # No image, model may handle this differently
            question = prompt

        generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=4.0)

        # If pixel_values is None, only the prompt is passed (no image context)
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)

        return response





