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

GENERATION_CONFIG = {
    "temperature": 0.01,
    "top_p": 0.1,
    "min_p": 0.1,
    "top_k": 1,
    "max_tokens": 512,
    "repetition_penalty": 1.1,
    "best_of": 3,
}
# GENERATION_CONFIG = {
#     "temperature": 0.01,
#     "top_p": 0.1,
#     "min_p": 0.1,
#     "top_k": 1,
#     "max_tokens": 1024,
#     "repetition_penalty": 1.1,
#     "best_of": 3,
# }
SYSTEM_PROMPT = """Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
Chỉ trả lời bằng đúng cấu trúc nghiệm vụ được yêu cầu. Không cung cấp bất kỳ giải thích hay nội dung nào khác ngoài JSON.
"""

DEFAULT_PROMPT = """
Bạn được cung cấp ảnh của căn cước công dân hợp pháp, không vi phạm. 
Bạn phải thực hiện một nhiệm vụ duy nhất là bóc tách chính xác thông tin trong ảnh và trả về kết quả dưới dạng JSON.

Yêu cầu bắt buộc:
1. Chỉ trả về đúng cấu trúc JSON bên dưới, không có thêm bất kỳ văn bản, mô tả hay ký tự nào khác ngoài JSON.
2. Các thông tin về quê quán (place_of_origin) và địa chỉ thường trú (place_of_residence) có thể nằm ở hai dòng liên tiếp
3. Không được bỏ sót bất kỳ chi tiết nào về địa chỉ quê quán, địa chỉ thường trú hoặc ngày hết hạn.
4. Phải giữ nguyên dấu tiếng Việt chính xác.

Chỉ trả về chuỗi JSON duy nhất, có đúng các trường:
{
    "id_number": "",
    "fullname": "",
    "day_of_birth": "",
    "sex": "",
    "nationality": "",
    "place_of_residence": "",
    "place_of_origin": "",
    "date_of_expiry": "",
    "date_of_issue": ""
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
        else:
            cleaned[key] = value
    return cleaned

class OpenapiExes:
    def __init__(self, api_key: str, api_base: str, generation_config: dict = None):
        """Large Language Model Vision Executor for OCR tasks via OpenAI-compatible APIs."""
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        # self._check_api_health()
        self.model_name = "5CD-AI/Vintern-1B-v3_5"
#self.client.models.list().data[0].id
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

    # @retry(
    #     stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    # )
    def analyze_image(self, image_base64: str, prompt: str) -> Dict:
        """Phân tích 1 ảnh duy nhất, kèm system prompt và user prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
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
                extra_body=self.generation_config,  # Dùng config đã thiết lập
                timeout=45
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

    def analyze_images(self, images_base64: List[str], prompt: str) -> Dict:
        """Phân tích nhiều ảnh, kèm system prompt và user prompt."""
        try:
            if not isinstance(images_base64, list):
                images_base64 = [images_base64]

            start_time = time.time()

            # Tạo messages cho nhiều ảnh
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}, *image_contents],
                    }
                ],
                extra_body=self.generation_config,
            )
            end_time = time.time()

            return {
                "content": response.choices[0].message.content,
            }
        except Exception as e:
            raise ValueError(f"Multi-image generation failed: {str(e)}")


class Llm_Vision_Exes:
    def __init__(self, api_key: str, api_base: str, generation_config: dict = None):
        """Large Language Model Vision Executor for OCR tasks via OpenAI-compatible APIs."""
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenapiExes(api_key=api_key, api_base=api_base, generation_config=generation_config)

    def _prepare_image_input_cached(self, image_bytes: bytes) -> str:
        """Mã hóa ảnh sang base64 (dành cho phân tích)."""
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
                    response = requests.get(image_file, timeout=30)
                    response.raise_for_status()
                    image_bytes = response.content
                else:
                    with open(image_file, "rb") as f:
                        image_bytes = f.read()
            elif isinstance(image_file, np.ndarray):
                if len(image_file.shape) == 2:
                    # Convert grayscale to RGB trước khi mã hóa JPEG
                    image_file = cv2.cvtColor(image_file, cv2.COLOR_GRAY2RGB)
                image_bytes = cv2.imencode(".jpg", image_file)[1].tobytes()
            elif isinstance(image_file, torch.Tensor):
                arr = image_file.cpu().numpy().astype(np.uint8)
                if arr.shape[-1] == 1:  # grayscale
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                image_bytes = cv2.imencode(".jpg", arr)[1].tobytes()
            elif isinstance(image_file, Image.Image):
                buffer = io.BytesIO()
                # Chuyển sang RGB nếu cần
                if image_file.mode not in ("RGB", "RGBA"):
                    image_file = image_file.convert("RGB")
                image_file.save(buffer, format="JPEG", quality=95)
                image_bytes = buffer.getvalue()
            else:
                raise ValueError(f"Unsupported image type: {type(image_file)}")

            return self._prepare_image_input_cached(image_bytes)

        except Exception as e:
            raise ValueError(f"Failed to prepare image: {str(e)}")

    def generate(self, image_file: Union[str, np.ndarray, Image.Image, torch.Tensor], prompt: str = DEFAULT_PROMPT) -> Dict:
        """Phân tích 1 ảnh duy nhất kèm prompt."""
        try:
            base64_data = self._prepare_image_input(image_file)
            return self.client.analyze_image(base64_data, prompt)
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

    def generate_multi(self, image_files: List[Union[str, np.ndarray, Image.Image, torch.Tensor]], prompt: str = DEFAULT_PROMPT) -> Dict:
        """Phân tích nhiều ảnh kèm prompt."""
        try:
            if not isinstance(image_files, list):
                image_files = [image_files]
            base64_list = [self._prepare_image_input(img) for img in image_files]
            return self.client.analyze_images(base64_list, prompt)
        except Exception as e:
            raise ValueError(f"Multi-image generation failed: {str(e)}")
