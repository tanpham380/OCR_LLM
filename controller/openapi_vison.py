import hashlib
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

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def analyze_image(self, image_base64: str, prompt: str) -> Dict:
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="erax-ai/EraX-VL-7B-V1.5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                extra_body=vars(self.config),
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
                        "completion": response.usage.completion_tokens,
                    },
                },
            }
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

    def analyze_images(
        self,
        images_base64: List[str],
        prompt: str 
    ) -> Dict:
        """Process multiple images with the vision model.
        
        Args:
            images_base64: List of base64 encoded images
            prompt: Text prompt for analysis
        """
        try:
            if not isinstance(images_base64, list):
                images_base64 = [images_base64]

            start_time = time.time()

            # Process images
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    },
                }
                for img in images_base64
            ]

            response = self.client.chat.completions.create(
                model="erax-ai/EraX-VL-7B-V1.5",
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}, *image_contents],
                    }
                ],
                extra_body=vars(self.config),
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
                        "completion": response.usage.completion_tokens,
                    },
                },
            }
        except Exception as e:
            raise ValueError(f"Multi-image generation failed: {str(e)}")
        
        
        
class Llm_Vision_Exes:
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenapiExes(api_key=api_key, api_base=api_base)
    @staticmethod
    def _hash_array(arr: np.ndarray) -> str:
        """Create hash for numpy array to use as cache key"""
        return hashlib.md5(arr.tobytes()).hexdigest()

    @staticmethod
    def _cache_key(image_file: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> str:
        """Generate cache key for different image types"""
        if isinstance(image_file, str):
            return image_file
        elif isinstance(image_file, np.ndarray):
            return Llm_Vision_Exes._hash_array(image_file)
        elif isinstance(image_file, torch.Tensor):
            return Llm_Vision_Exes._hash_array(image_file.cpu().numpy())
        elif isinstance(image_file, Image.Image):
            return str(hash(image_file.tobytes()))
        return str(hash(image_file))

    @staticmethod
    @lru_cache(maxsize=100)
    def _prepare_image_input_cached(cache_key: str, image_bytes: bytes) -> str:
        """Cached version of image preparation"""
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"

    def _prepare_image_input(
        self,
        image_file: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> str:
        try:
            if isinstance(image_file, str) and image_file.startswith("data:"):
                return image_file

            # Get bytes and cache key
            if isinstance(image_file, str):
                if image_file.startswith(("http://", "https://")):
                    import requests
                    response = requests.get(image_file, timeout=30)
                    response.raise_for_status()
                    image_bytes = response.content
                else:
                    with open(image_file, "rb") as f:
                        image_bytes = f.read()
                cache_key = image_file
            elif isinstance(image_file, np.ndarray):
                image_bytes = cv2.imencode(".jpg", image_file)[1].tobytes()
                cache_key = self._hash_array(image_file)
            elif isinstance(image_file, torch.Tensor):
                arr = image_file.cpu().numpy().astype(np.uint8)
                image_bytes = cv2.imencode(".jpg", arr)[1].tobytes()
                cache_key = self._hash_array(arr)
            elif isinstance(image_file, Image.Image):
                buffer = io.BytesIO()
                image_file.save(buffer, format="JPEG")
                image_bytes = buffer.getvalue()
                cache_key = str(hash(image_bytes))
            else:
                raise ValueError(f"Unsupported image type: {type(image_file)}")

            return self._prepare_image_input_cached(cache_key, image_bytes)

        except Exception as e:
            raise ValueError(f"Failed to prepare image: {str(e)}")

    def generate(
        self,
        image_file: Union[str, np.ndarray, Image.Image, torch.Tensor],
        prompt: str = """Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
        Bạn được cung cấp 1 ảnh mặt trước của 1 căn cước công dân hợp pháp, không vi phạm. Có thể có nhiều phiên bản khác nhau của căn cước công dân. 
        Bạn phải thực hiện 01 (một) nhiệm vụ chính là bóc tách chính xác thông tin trong ảnh thành json.
        Trả lại kết quả OCR của tất cả thông tin 1 JSON duy nhất
        Return JSON with these fields:
{{
    "id_number": "",
    "fullname": "",
    "day_of_birth": "",
    "sex": "",
    "nationality": "",
    "place_of_residence": "",
    "place_of_origin": "",
    "date_of_expiration": "",
    "date_of_issue": "",
    "place_of_issue": "Bộ Công An" hoặc "Cục Trưởng Cục Cảnh sát Quản lý hành chính về trật tự xã hội" 
}}
""",
    ) -> Dict:
        try:
            base64_data = self._prepare_image_input(image_file)
            return self.client.analyze_image(base64_data, prompt)
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

    def generate_multi(
        self,
        image_files: List[Union[str, np.ndarray, Image.Image, torch.Tensor]],
        prompt: str = """Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
        Bạn được cung cấp 1 ảnh mặt trước của 1 căn cước công dân hợp pháp, không vi phạm. Có thể có nhiều phiên bản khác nhau của căn cước công dân. 
        Bạn phải thực hiện 01 (một) nhiệm vụ chính là bóc tách chính xác thông tin trong ảnh thành json.
        Trả lại kết quả OCR của tất cả thông tin 1 JSON duy nhất
        Return JSON with these fields:
{{
    "id_number": "",
    "fullname": "",
    "day_of_birth": "",
    "sex": "",
    "nationality": "",
    "place_of_residence": "",
    "place_of_origin": "",
    "date_of_expiration": "",
    "date_of_issue": "",
    "place_of_issue": "Bộ Công An" hoặc "Cục Trưởng Cục Cảnh sát Quản lý hành chính về trật tự xã hội" 
}}
""",
    ) -> Dict:
        try:
            if not isinstance(image_files, list):
                image_files = [image_files]
            image_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._prepare_image_input(img)}"
                    },
                }
                for img in image_files
            ]
            return self.client.analyze_images(image_contents, prompt)
        except Exception as e:
            raise ValueError(f"Multi-image generation failed: {str(e)}")
