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

import requests

# from app_utils.util import clean_content



GENERATION_CONFIG = {
    "temperature": 0.01,
    "top_p": 0.1,
    "min_p": 0.1,
    "top_k": 1,
    "max_tokens": 1024,
    "repetition_penalty": 1.1,
    "best_of": 3,
}

#     "id_number": {
#         "vi": ["Số","Số định danh cá nhân"],
#         "en": ["No.", "Personal Identification Number", "ID Number"],
#         "format": "12 chữ số "
#     },
#     "fullname": {
#         "vi": ["Họ và tên", "Họ, chữ đệm và tên khai sinh"],
#         "en": ["Full name"],
#     },
#     "day_of_birth": {
#         "vi": ["Ngày sinh", "Ngày, tháng, năm sinh"],
#         "en": ["Date of birth"],
#         "format": "DD/MM/YYYY"
#     },
#     "sex": {
#         "vi": ["Giới tính"],
#         "en": ["Sex"],
#         "valid": ["Nam", "Nữ"]
#     },
#     "nationality": {
#         "vi": ["Quốc tịch"],
#         "en": ["Nationality"],
#         "default": "Việt Nam"
#     },
#     "place_of_residence": {
#         "vi": ["Nơi thường trú", "Nơi cư trú],
#         "en": ["Place of residence"],
#     },
#     "place_of_origin": {
#         "vi": ["Quê quán", "Nơi đăng kí khai sinh"],
#         "en": ["Place of origin", "Place of birth"],
#     },
#     "date_of_expiration": {
#         "vi": ["Ngày, tháng, năm", "Ngày,Tháng, Năm hết hạn"],
#         "en": ["Date of expiry" , "Date, month, year:],
#         "format": "DD/MM/YYYY"
#     },
#     "date_of_issue": {
#         "vi": ["Ngày cấp", "Ngày,Tháng, Năm cấp"],
#         "en": ["Date of issue"],
#         "format": "DD/MM/YYYY"
#     },
#     "place_of_issue": {
#         "vi": ["Nơi cấp", "Cơ quan cấp"],
#         "en": ["Place of issue"],
#     }
# }
# - Nơi cấp chỉ có 2 giá trị là "Cục trưởng Cục Cảnh sát Quản lý Hành chính về Trật tự Xã hội" hoặc "Bộ Công An".
    # "place_of_issue": ""

DEFAULT_PROMPT = """
Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
Bạn được cung cấp ảnh của căn cước công dân hợp pháp, không vi phạm. Có thể có nhiều phiên bản khác nhau của căn cước công dân.
Bạn phải thực hiện 01 (một) nhiệm vụ chính là bóc tách chính xác thông tin trong ảnh thành json theo qui tắc sau.
- Các thông tin quê quán và dịa chỉ thường trú có thể nằm ở 2 dòng liên tiếp nhau. 
- Không được bỏ sót bất kỳ thông tin chi tiết nào về địa chỉ quê quán hoặc địa chỉ thường trú hoặc ngày hết hạn của thẻ.
- Bảo đảm các câu từ có dấu tiếng Việt là đầy đủ và chính xác.
- Trả lại kết quả OCR duy nhất với các trường sau không thêm gì khác Ngoài:
{
    "id_number": "",
    "fullname": "",
    "day_of_birth": "",
    "sex": "",
    "nationality": "",
    "place_of_residence": "",
    "place_of_origin": "", 
    "date_of_expiration": "",
    "date_of_issue": "",
}
"""

def clean_response_content(response_content: str) -> dict:
    import json    
    if isinstance(response_content, str):
        response_content = json.loads(response_content)        
    cleaned = {}
    for key, value in response_content.items():
        if isinstance(value, str):
            cleaned[key] = value.replace("\\n", " ").replace("\n", " ").strip()
        elif isinstance(value, dict):
            cleaned[key] = clean_response_content(value)
    
    return cleaned

class OpenapiExes:
    def __init__(self, api_key: str, api_base: str , generation_config: dict = None):
        """Large Language Model Vision Executor for OCR tasks.
    
        This class provides an interface to process images using Vision-Language models
        through OpenAI-compatible APIs.

        Args:
            api_key (str): API key for authentication
            api_base (str): Base URL for the API endpoint
            generation_config (dict, optional): Configuration parameters for text generation.
                Supported parameters:
                    - temperature (float): Controls randomness (0.0-1.0). Default: 0.01
                    - top_p (float): Nucleus sampling parameter (0.0-1.0). Default: 0.1
                    - min_p (float): Minimum probability for tokens. Default: 0.1
                    - top_k (int): Number of highest probability tokens. Default: 1
                    - max_tokens (int): Maximum tokens to generate. Default: 1024
                    - repetition_penalty (float): Penalty for repeated tokens. Default: 1.1
                    - best_of (int): Number of completion options. Default: 3

        Example:
            >>> config = {
            ...     "temperature": 0.1,
            ...     "max_tokens": 2048,
            ...     "best_of": 3
            ... }
            >>> ocr = Llm_Vision_Exes(
            ...     api_key="your-key",
            ...     api_base="http://api.example.com",
            ...     generation_config=config
            ... )
        """
        # Remove trailing slash if present
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        self._check_api_health()
        self.model_name = self.client.models.list().data[0].id
        self.generation_config = generation_config or GENERATION_CONFIG


    def get_instant_api(self):
        return self.client

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _check_api_health(self) -> bool:
        """Check if API is accessible and responding via /health endpoint"""
        try:
            base_url = self.api_base.rsplit("/v1", 1)[0]
            health_url = f"{base_url}/health"

            response = requests.get(
                health_url,
                timeout=15,
                verify=False, 
            )

            if response.status_code != 200:
                raise ConnectionError(
                    f"Health check failed with status {response.status_code}"
                )

            return True

        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to API at {self.api_base}: {str(e)}"
            )
        except Exception as e:
            raise ConnectionError(f"Health check failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def analyze_image(self, image_base64: str, prompt: str) -> Dict:
        try:
            # start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{image_base64}"},
                            },
                        ],
                    }
                ],
                extra_body=self.generation_config,  # Use updated config
            )
            # end_time = time.time()
            content = clean_response_content(response.choices[0].message.content)
            # if isinstance(content, dict):
            #     for key, value in content.items():
            #         if isinstance(value, str):
            #             content[key] = value.replace("\n", " ")
            return {
                "content": content,
                # "metadata": {
                #     "model": response.model,
                #     "created": response.created,
                #     "response_time": f"{end_time - start_time:.2f}",
                #     "tokens": {
                #         "total": response.usage.total_tokens,
                #         "prompt": response.usage.prompt_tokens,
                #         "completion": response.usage.completion_tokens,
                #     },
                # },
            }
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

    def analyze_images(self, images_base64: List[str], prompt: str) -> Dict:
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
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                }
                for img in images_base64
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}, *image_contents],
                    }
                ],
                extra_body=self.generation_config,  # Use updated config
            )
            end_time = time.time()

            

            return {
                "content": response.choices[0].message.content,

            }
        except Exception as e:
            raise ValueError(f"Multi-image generation failed: {str(e)}")


class Llm_Vision_Exes:
    def __init__(self, api_key: str, api_base: str ,  generation_config: dict = None):
        """Large Language Model Vision Executor for OCR tasks.
    
        This class provides an interface to process images using Vision-Language models
        through OpenAI-compatible APIs.

        Args:
            api_key (str): API key for authentication
            api_base (str): Base URL for the API endpoint
            generation_config (dict, optional): Configuration parameters for text generation.
                Supported parameters:
                    - temperature (float): Controls randomness (0.0-1.0). Default: 0.01
                    - top_p (float): Nucleus sampling parameter (0.0-1.0). Default: 0.1
                    - min_p (float): Minimum probability for tokens. Default: 0.1
                    - top_k (int): Number of highest probability tokens. Default: 1
                    - max_tokens (int): Maximum tokens to generate. Default: 1024
                    - repetition_penalty (float): Penalty for repeated tokens. Default: 1.1
                    - best_of (int): Number of completion options. Default: 3

        Example:
            >>> config = {
            ...     "temperature": 0.1,
            ...     "max_tokens": 2048,
            ...     "best_of": 3
            ... }
            >>> ocr = Llm_Vision_Exes(
            ...     api_key="your-key",
            ...     api_base="http://api.example.com",
            ...     generation_config=config
            ... )
        """
        self.api_key = api_key
        self.api_base = api_base

        self.client = OpenapiExes(api_key=api_key, api_base=api_base , generation_config = generation_config)



    def _prepare_image_input_cached(self,image_bytes: bytes) -> str:
        """Cached version of image preparation"""
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"

    def _prepare_image_input(
            self, image_file: Union[str, np.ndarray, Image.Image, torch.Tensor]
        ) -> str:
            try:
                if isinstance(image_file, str) and image_file.startswith("data:"):
                    return image_file
                if isinstance(image_file, str):
                    if image_file.startswith(("http://", "https://")):
                        import requests
                        response = requests.get(image_file, timeout=30)
                        response.raise_for_status()
                        image_bytes = response.content
                    else:
                        with open(image_file, "rb") as f:
                            image_bytes = f.read()
                elif isinstance(image_file, np.ndarray):
                    if len(image_file.shape) == 2:
                        # Convert grayscale to RGB before JPEG encoding
                        image_file = cv2.cvtColor(image_file, cv2.COLOR_GRAY2RGB)
                    image_bytes = cv2.imencode(".jpg", image_file)[1].tobytes()
                elif isinstance(image_file, torch.Tensor):
                    arr = image_file.cpu().numpy().astype(np.uint8)
                    if arr.shape[-1] == 1:  # Handle grayscale
                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                    image_bytes = cv2.imencode(".jpg", arr)[1].tobytes()
                elif isinstance(image_file, Image.Image):
                    buffer = io.BytesIO()
                    # Convert to RGB if needed
                    if image_file.mode not in ('RGB', 'RGBA'):
                        image_file = image_file.convert('RGB')
                    image_file.save(buffer, format='JPEG', quality=95)
                    image_bytes = buffer.getvalue()
                else:
                    raise ValueError(f"Unsupported image type: {type(image_file)}")

                return self._prepare_image_input_cached(image_bytes)

            except Exception as e:
                raise ValueError(f"Failed to prepare image: {str(e)}")

    def generate(
        self,
        image_file: Union[str, np.ndarray, Image.Image, torch.Tensor],
        prompt: str = DEFAULT_PROMPT
    ) -> Dict:
        try:
            base64_data = self._prepare_image_input(image_file)
            return self.client.analyze_image(base64_data, prompt)
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

    def generate_multi(
        self,
        image_files: List[Union[str, np.ndarray, Image.Image, torch.Tensor]],
        prompt: str = DEFAULT_PROMPT
    ) -> Dict:
        try:
            if not isinstance(image_files, list):
                image_files = [image_files]
            return self.client.analyze_images(
                [self._prepare_image_input(img) for img in image_files], prompt
            )
        except Exception as e:
            raise ValueError(f"Multi-image generation failed: {str(e)}")