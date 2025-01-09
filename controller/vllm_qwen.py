import io
import os
import base64
import json
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, Dict, Any
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class VLLM_Exes:
    def __init__(
        self,
        model_path: str = "erax-ai/EraX-VL-7B-V1.5",
    ):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        
        # Default generation config
        self.generation_config = self.model.generation_config
        self.generation_config.do_sample = True
        self.generation_config.temperature = 0.01
        self.generation_config.top_k = 1
        self.generation_config.top_p = 0.1
        self.generation_config.min_p = 0.1
        self.generation_config.best_of = 5
        self.generation_config.max_new_tokens = 2048
        
    def generate_multi(
        self,
        image_files: Union[str, list, np.ndarray, Image.Image, torch.Tensor],
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
    "place_of_issue": ""
}}
""",
    ) -> str:
        """Generate text from multiple images and prompt.
        
        Args:
            image_files: Single image or list of images (file paths, URLs, arrays, PIL Images or tensors)
            prompt: Text prompt to guide generation
            
        Returns:
            str: Generated text response
        """
        try:
            # Handle single image case
            if not isinstance(image_files, list):
                image_files = [image_files]
                
            # Process all images to base64
            image_contents = []
            for img in image_files:
                base64_data = self._prepare_image_input(img)
                image_contents.append({
                    "type": "image",
                    "image": base64_data
                })
                
            # Add text prompt
            image_contents.append({
                "type": "text", 
                "text": prompt
            })
            
            # Create messages structure
            messages = [{
                "role": "user",
                "content": image_contents
            }]
            
            # Process inputs
            tokenized_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Create model inputs
            inputs = self.processor(
                text=[tokenized_text],
                images=image_inputs, 
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            
            # Generate output
            generated_ids = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
        except Exception as e:
            raise ValueError(f"Multi-image generation failed: {str(e)}")
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
    "place_of_issue": ""
}}
""",
    ) -> str:
        """Generate text from image and prompt.
        
        Args:
            image_file: Image input as file path, URL, numpy array, PIL Image or tensor
            prompt: Text prompt to guide generation
            
        Returns:
            str: Generated text response
        """
        try:
            # Prepare image input
            base64_data = self._prepare_image_input(image_file)
            
            # Create messages
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": base64_data},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Process inputs
            tokenized_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Create model inputs
            inputs = self.processor(
                text=[tokenized_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            
            # Generate output
            generated_ids = self.model.generate(
                **inputs, 
                generation_config=self.generation_config
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")
        
            
    def _prepare_image_input(
            self,
            image_file: Union[str, np.ndarray, Image.Image, torch.Tensor],
        ) -> str:
            try:
                if isinstance(image_file, str) and image_file.startswith("data:"):
                    return image_file

                # Get image bytes
                if isinstance(image_file, str):
                    if image_file.startswith("http"):
                        import requests
                        from time import sleep
                        
                        retry_count = 3
                        last_error = None
                        
                        for attempt in range(retry_count):
                            try:
                                response = requests.get(image_file, timeout=30)
                                response.raise_for_status()
                                image_bytes = response.content
                                break
                            except Exception as e:
                                last_error = e
                                if attempt < retry_count - 1:
                                    sleep(1 * (attempt + 1))
                                else:
                                    raise ValueError(f"Failed to download image after {retry_count} attempts: {str(last_error)}")
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

                # Return with data URI prefix
                base64_str = base64.b64encode(image_bytes).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_str}"
                
            except Exception as e:
                raise ValueError(f"Failed to prepare image: {str(e)}")