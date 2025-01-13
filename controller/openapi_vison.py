import io
import os
import base64
import json
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import time
import logging
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenapiExes:
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

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
                extra_body={
                    "temperature": 0.01,
                    "top_p": 0.1,
                    "min_p": 0.1,
                    "top_k": 1,
                    "max_tokens": 1024,
                    "repetition_penalty": 1.1,
                    "best_of": 5,
                    "use_beam_search": False
                }
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

    def _prepare_image_input(self, image_file: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> str:
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
                    with open(image_file, 'rb') as f:
                        image_bytes = f.read()
            elif isinstance(image_file, np.ndarray):
                image_bytes = cv2.imencode('.jpg', image_file)[1].tobytes()
            elif isinstance(image_file, torch.Tensor):
                arr = image_file.cpu().numpy().astype(np.uint8)
                image_bytes = cv2.imencode('.jpg', arr)[1].tobytes()
            elif isinstance(image_file, Image.Image):
                buffer = io.BytesIO()
                image_file.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
            else:
                raise ValueError(f"Unsupported image type: {type(image_file)}")

            base64_str = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_str}"
        except Exception as e:
            raise ValueError(f"Failed to prepare image: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, image_file: Union[str, np.ndarray, Image.Image, torch.Tensor], prompt: str) -> Dict:
        try:
            base64_data = self._prepare_image_input(image_file)
            return self.client.analyze_image(base64_data, prompt)
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_multi(self, image_files: List[Union[str, np.ndarray, Image.Image, torch.Tensor]], prompt: str) -> Dict:
        try:
            if not isinstance(image_files, list):
                image_files = [image_files]

            image_contents = []
            for img in image_files:
                base64_data = self._prepare_image_input(img)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": base64_data}
                })

            start_time = time.time()
            response = self.client.chat.completions.create(
                model="erax-ai/EraX-VL-7B-V1.5",
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_contents
                    ]
                }],
                extra_body={
                    "temperature": 0.01,
                    "top_p": 0.1,
                    "min_p": 0.1, 
                    "top_k": 1,
                    "max_tokens": 1024,
                    "repetition_penalty": 1.1,
                    "best_of": 5,
                    "use_beam_search": False
                }
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