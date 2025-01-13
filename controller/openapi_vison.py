import io
import base64
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, Dict, Any, List
import time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class ModelConfig:
    """Default model configuration"""
    temperature: float = 0.01
    top_p: float = 0.1
    min_p: float = 0.1
    top_k: int = 1
    max_tokens: int = 1024
    repetition_penalty: float = 1.1
    best_of: int = 5
    use_beam_search: bool = False

class OpenapiExes:
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        self.config = ModelConfig()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_image(self, image_base64: str, prompt: str) -> Dict:
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="erax-ai/EraX-VL-7B-V1.5",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }],
                extra_body=vars(self.config)
            )
            end_time = time.time()
            
            return {
                "content": response.choices[0].message.content,
                "metadata": {
                    "model": response.model,
                    "created": response.created,
                    "response_time": f"{end_time - start_time:.2f}",
                    "tokens": {
                        "total": response.usage.total_tokens,
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens
                    }
                }
            }
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

class Llm_Vision_Exes:
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenapiExes(api_key=api_key, api_base=api_base)

    @staticmethod
    @lru_cache(maxsize=100)
    def _prepare_image_input(image_file: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> str:
        """Convert image to base64 with caching for better performance"""
        try:
            if isinstance(image_file, str) and image_file.startswith("data:"):
                return image_file

            def get_image_bytes() -> bytes:
                if isinstance(image_file, str):
                    if image_file.startswith(("http://", "https://")):
                        import requests
                        response = requests.get(image_file, timeout=30)
                        response.raise_for_status()
                        return response.content
                    return open(image_file, 'rb').read()
                elif isinstance(image_file, np.ndarray):
                    return cv2.imencode('.jpg', image_file)[1].tobytes()
                elif isinstance(image_file, torch.Tensor):
                    arr = image_file.cpu().numpy().astype(np.uint8)
                    return cv2.imencode('.jpg', arr)[1].tobytes()
                elif isinstance(image_file, Image.Image):
                    buffer = io.BytesIO()
                    image_file.save(buffer, format='JPEG')
                    return buffer.getvalue()
                raise ValueError(f"Unsupported image type: {type(image_file)}")

            image_bytes = get_image_bytes()
            base64_str = base64.b64encode(image_bytes).decode('utf-8')
            return base64_str

        except Exception as e:
            raise ValueError(f"Failed to prepare image: {str(e)}")

    def generate(self, image_file: Union[str, np.ndarray, Image.Image, torch.Tensor], prompt: str) -> Dict:
        try:
            base64_data = self._prepare_image_input(image_file)
            return self.client.analyze_image(base64_data, prompt)
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

    def generate_multi(self, image_files: List[Union[str, np.ndarray, Image.Image, torch.Tensor]], prompt: str) -> Dict:
        try:
            if not isinstance(image_files, list):
                image_files = [image_files]

            start_time = time.time()
            
            # Process all images in parallel using list comprehension
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._prepare_image_input(img)}"}
                }
                for img in image_files
            ]

            response = self.client.chat.completions.create(
                model="erax-ai/EraX-VL-7B-V1.5",
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_contents
                    ]
                }],
                extra_body=vars(self.client.config)
            )
            end_time = time.time()

            return {
                "content": response.choices[0].message.content,
                "metadata": {
                    "model": response.model,
                    "created": response.created,
                    "response_time": f"{end_time - start_time:.2f}",
                    "tokens": {
                        "total": response.usage.total_tokens,
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens
                    }
                }
            }
        except Exception as e:
            raise ValueError(f"Multi-image generation failed: {str(e)}")