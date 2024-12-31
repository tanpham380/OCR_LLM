import io
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import cv2
from PIL import Image
import math
from functools import lru_cache
from typing import Union, List, Tuple, Optional, Set
import concurrent.futures
import base64

class EraXVLOcrModel:
    def __init__(self, 
                model_path: str = "erax-ai/EraX-VL-2B-V1.5",
                device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else \
                     torch.device(device) if isinstance(device, str) else device
        
        self.force_single_gpu = device is not None
        self.model_path = model_path
        
        # Move constants to device
        self.register_constants()
        
        # Initialize model with optimized settings
        self.initialize_model()
        
        # Prepare transform pipeline and aspect ratio cache
        self._transform_cache = {}
        self._target_ratios_cache = {}
        
        
    def register_constants(self):
        self.default_prompt = (
            """Hãy trích xuất thông tin từ căn cước công dân trong ảnh với cấu trúc sau:
            {
                "id_number": "Số CCCD",
                "fullname": "Họ và tên / Full name",
                "day_of_birth": "Ngày sinh / Date of birth", 
                "sex": "Giới tính / Sex",
                "nationality": "Quốc tịch / Nationality",
                "place_of_residence": "Nơi thường trú / Place of residence",
                "place_of_origin": "Quê quán / Place of origin",
                "date_of_expiration": "Ngày hết hạn / Date of expiry",
                "date_of_issue": "Ngày cấp (nếu có)",
                "place_of_issue": "Nơi cấp (nếu có)"
            }
            HƯỚNG DẪN XỬ LÝ FULLNAME:
            - Quy tắc với fullname:
               - Viết HOA toàn bộ
               - Kiểm tra từng chữ với danh sách mẫu
               - Không dùng dấu sai dù chỉ một chữ
            - Chú ý đặc biệt đến chữ in hoa có dấu. Không được nhầm:
                + "Bạch" thành "Bàch"
                + "Ể" thành "Ề"
                + Tương tự với các âm Ê, Ế, Ề, Ể, Ễ, Ệ khác
            - Quy tắc kiểm tra dấu:
                KHÔNG DẤU: A, E, I, O, U
                HUYỀN: À, È, Ì, Ò, Ù
                SẮC: Á, É, Í, Ó, Ú  
                HỎI: Ả, Ẻ, Ỉ, Ỏ, Ủ
                NGÃ: Ã, Ẽ, Ĩ, Õ, Ũ
                NẶNG: Ạ, Ẹ, Ị, Ọ, Ụ
            - Không cần khôi phục dữ liệu nếu chữ đã sai, mà hãy cố gắng nhận diện và giữ đúng chính tả ngay từ đầu.
            - Các tỉnh, thành phố ở Việt Nam (nếu có) vẫn giữ nguyên tên đầy đủ, bao gồm dấu tiếng Việt chính xác.

            Yêu cầu QUAN TRỌNG:
            1. Trả về định dạng json. Không diễn giải cách làm, không tóm tắt, chỉ trả lại duy nhất 1 json.
            2. Với place_of_origin và place_of_residence:
                - PHẢI ghi đầy đủ toàn bộ địa chỉ
                - KHÔNG ĐƯỢC rút gọn hay cắt bớt bất kỳ phần nào
            3. Trường hợp không tìm thấy thông tin thì để null
            4. KHÔNG thêm bất kỳ chú thích hay giải thích nào
            5. KHÔNG bọc kết quả trong key 'json' hay key khác
            6. Các trường khác giữ nguyên định dạng gốc
            """
        )

    def initialize_model(self):
        """Initialize model with optimized settings"""
        device_map = str(self.device) if self.force_single_gpu else "auto"
        
        # max_memory = {i: "18GB" for i in range(torch.cuda.device_count())}
        # max_memory["cpu"] = "30GB"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            attn_implementation="eager", # replace with "flash_attention_2" if your GPU is Ampere architecture

            # max_memory=max_memory
        ).eval()

        # self.vision_device = (
        #     f"cuda:{device_map.get('transformer.visual', 0)}"
        #     if isinstance(device_map, dict) and torch.cuda.is_available()
        #     else str(self.device)
        # )
        
        # Initialize tokenizer with optimized settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            model_max_length=8196
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )

    


    def _prepare_image_input(self, image_file):
        """Convert image to base64 format"""
        if isinstance(image_file, str):
            if image_file.startswith('data:'):
                return image_file
            with open(image_file, "rb") as f:
                encoded_image = base64.b64encode(f.read())
            return f"data:image;base64,{encoded_image.decode('utf-8')}"
        
        # Handle numpy array, PIL Image, or torch tensor
        if isinstance(image_file, (np.ndarray, Image.Image, torch.Tensor)):
            img_byte_arr = io.BytesIO()
            if isinstance(image_file, np.ndarray):
                Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)).save(img_byte_arr, format='PNG')
            elif isinstance(image_file, Image.Image):
                image_file.save(img_byte_arr, format='PNG')
            else:  # torch.Tensor
                Image.fromarray(image_file.cpu().numpy().astype(np.uint8)).save(img_byte_arr, format='PNG')
            encoded_image = base64.b64encode(img_byte_arr.getvalue())
            return f"data:image;base64,{encoded_image.decode('utf-8')}"
        raise ValueError("Unsupported image format")
    @torch.no_grad()
    def process_image(self,
                     image_file: Optional[Union[np.ndarray, Image.Image, torch.Tensor, str]] = None,
                     custom_prompt: Optional[str] = None,
                     input_size: int = 448,
                     max_num: int = 6) -> str:
        """
        Process image and generate text response
        """
        try:
            torch.cuda.empty_cache()
            prompt = custom_prompt if custom_prompt else self.default_prompt
            base64_data = self._prepare_image_input(image_file)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_data},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            tokenized_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[tokenized_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Updated generation parameters
            generation_config = {
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 1,
                "top_p": 0.9,
                "min_p": 0.1,
                "max_new_tokens": 2048,
                "repetition_penalty": 1.06
            }

            generated_ids = self.model.generate(
                **inputs,
                **generation_config
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            print(response)
            return response

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise
        
        finally:
            torch.cuda.empty_cache()