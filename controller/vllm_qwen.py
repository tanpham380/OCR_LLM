import base64
import torch
from vllm import LLM, SamplingParams
from functools import lru_cache
from typing import Optional, Union, Dict, Any, List
from transformers import AutoProcessor
import math
import base64
import numpy as np
import io
import cv2
from PIL import Image
import torch

class VLLM_Exes:
    @staticmethod
    def get_default_sampling_params():
        return {
            'temperature': 0.01,
            'top_p': 0.1,
            'min_p': 0.1,
            'top_k': 1,
            'max_tokens': 1024,
            'repetition_penalty': 1.16,
            'best_of': 5
        }

    @staticmethod
    @lru_cache(maxsize=8)
    def get_device_map(num_gpus: int) -> Dict[str, int]:
        if num_gpus <= 1:
            return {"cuda:0": 1}
        return {f"cuda:{i}": math.ceil(100/num_gpus) for i in range(num_gpus)}

    def __init__(
        self,
        model_name: str = "erax-ai/EraX-VL-2B-V1.5",
        tensor_parallel_size: Optional[int] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        gpu_memory_utilization: float = 0.85,
        max_seq_len: int = 2048,
        quantization: Optional[str] = "awq",
        dtype: str = "float16",
        swap_space: int = 4,
        max_num_seqs: int = 5,
        **kwargs
    ):
        # GPU setup
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.num_gpus = torch.cuda.device_count()
        self.tensor_parallel_size = tensor_parallel_size or self.num_gpus
        
        # Sampling parameters
        sample_params = self.get_default_sampling_params()
        if sampling_params:
            sample_params.update(sampling_params)
        self.sampling_params = SamplingParams(**sample_params)

        # Model configuration
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_seq_len_to_capture=max_seq_len,
            trust_remote_code=True,
            dtype=dtype,
            quantization=quantization,
            enforce_eager=self.tensor_parallel_size == 1,
            swap_space=swap_space,
            disable_custom_all_reduce=self.tensor_parallel_size == 1,
            max_num_seqs=max_num_seqs,
            **kwargs
        )        
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, Dict],
        image_files: Optional[List[str]] = None,
        system_prompt: str = "You are a helpful assistant.",
        custom_sampling_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response for text or image+text input.
        
        Args:
            prompt: Text prompt or question
            image_files: Optional list of image file paths
            system_prompt: System prompt for chat
            custom_sampling_params: Optional sampling parameters
        """
        # Handle sampling parameters
        sampling_params = self.sampling_params
        if custom_sampling_params:
            sampling_params = SamplingParams(**custom_sampling_params)

        # Process images if provided
        if image_files:
            image_data = [self._prepare_image_input(img) for img in image_files]
            messages = [{
                "role": "system", 
                "content": system_prompt
            }, {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in image_data],
                    {"type": "text", "text": prompt}
                ]
            }]
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Generate response
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def _prepare_image_input(self, image_file: str) -> str:
        with open(image_file, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        return base64_data


    def _prepare_image_input(self, image_file: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> str:
        """Convert image to base64 format.
        
        Args:
            image_file: Can be:
                - str: File path or data URI
                - np.ndarray: NumPy array image
                - PIL.Image: PIL Image
                - torch.Tensor: Torch tensor image
                
        Returns:
            str: Base64 encoded image with data URI prefix
        """
        try:
            # Handle data URI
            if isinstance(image_file, str):
                if image_file.startswith('data:'):
                    return image_file
                # Handle file path
                with open(image_file, "rb") as f:
                    encoded_image = base64.b64encode(f.read())
                return f"data:image;base64,{encoded_image.decode('utf-8')}"
            
            # Handle numpy array
            if isinstance(image_file, np.ndarray):
                img_pil = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))
            # Handle torch tensor
            elif isinstance(image_file, torch.Tensor):
                img_pil = Image.fromarray(image_file.numpy().astype(np.uint8))
            # Handle PIL Image
            elif isinstance(image_file, Image.Image):
                img_pil = image_file
            else:
                raise ValueError(f"Unsupported image type: {type(image_file)}")
                
            # Convert to base64
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            encoded_image = base64.b64encode(img_byte_arr.getvalue())
            return f"data:image;base64,{encoded_image.decode('utf-8')}"
            
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")
    
    
    @classmethod
    def create_optimized(cls, gpu_tier: str = "mid") -> 'VLLM_Exes':
        """Factory method for pre-configured optimizations"""
        configs = {
            "low": {
                "gpu_memory_utilization": 0.7,
                "max_seq_len": 1024,
                "swap_space": 2,
                "tensor_parallel_size": 1
            },
            "mid": {
                "gpu_memory_utilization": 0.85,
                "max_seq_len": 2048,
                "swap_space": 4,
                "tensor_parallel_size": None
            },
            "high": {
                "gpu_memory_utilization": 0.95,
                "max_seq_len": 4096,
                "swap_space": 8,
                "tensor_parallel_size": None
            }
        }
        return cls(**configs.get(gpu_tier, configs["mid"]))