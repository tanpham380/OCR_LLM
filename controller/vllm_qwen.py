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
        model_path: str = "erax-ai/EraX-VL-2B-V1.5",
    ):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
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
        self.generation_config.temperature = 1.0
        self.generation_config.top_k = 1
        self.generation_config.top_p = 0.9
        self.generation_config.min_p = 0.1
        self.generation_config.best_of = 5
        self.generation_config.max_new_tokens = 2048
        self.generation_config.repetition_penalty = 1.06

    def generate(
        self,
        image_file: Union[str, np.ndarray, Image.Image, torch.Tensor],
        prompt: str = "Trích xuất thông tin nội dung từ hình ảnh được cung cấp.",
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

            # Get initial image
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
                            img = Image.open(io.BytesIO(response.content))
                            break
                        except Exception as e:
                            last_error = e
                            if attempt < retry_count - 1:
                                sleep(1 * (attempt + 1))
                            else:
                                raise ValueError(f"Failed to download image after {retry_count} attempts: {str(last_error)}")
                else:
                    img = Image.open(image_file)
            elif isinstance(image_file, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))
            elif isinstance(image_file, torch.Tensor):
                arr = image_file.cpu().numpy().astype(np.uint8)
                img = Image.fromarray(arr)
            elif isinstance(image_file, Image.Image):
                img = image_file
            else:
                raise ValueError(f"Unsupported image type: {type(image_file)}")

            # Ensure RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Save to buffer with compression
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')  # Added default quality
            buffer.seek(0)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{encoded}"

        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")