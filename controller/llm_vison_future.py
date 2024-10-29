import torch
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import cv2
from PIL import Image
import math
from functools import lru_cache
from typing import Union, List, Tuple, Optional, Set
import concurrent.futures

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

@lru_cache(maxsize=8)
def get_device_map(model_name: str, world_size: int) -> Union[str, dict]:
    """Cached device mapping calculation"""
    if world_size == 1:
        return "cuda:0"
        
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32,
        'InternVL2-8B': 32, 'InternVL2-26B': 48, 'InternVL2-40B': 60,
        'InternVL2-Llama3-76B': 80, "Vintern-3B-beta": 36
    }[model_name]
    
    device_map = {}
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    layers_distribution = [num_layers_per_gpu] * world_size
    layers_distribution[0] = math.ceil(layers_distribution[0] * 0.5)
    
    # Pre-define components
    base_components = {
        'vision_model': 0, 'mlp1': 0,
        'language_model.model.tok_embeddings': 0,
        'language_model.model.embed_tokens': 0,
        'language_model.output': 0,
        'language_model.model.norm': 0,
        'language_model.lm_head': 0,
        'language_model.model.rotary_emb': 0,
        'language_model.model.wte': 0,
        'language_model.model.ln_f': 0
    }
    device_map.update(base_components)
    
    # Map layers
    layer_cnt = 0
    for i, num_layer in enumerate(layers_distribution):
        for _ in range(num_layer):
            if layer_cnt < num_layers:
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
    
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map

class VinternOCRModel:
    def __init__(self, 
                model_path: str = "app_utils/weights/Vintern-3B-beta",
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
        """Register constants on the appropriate device"""
        self.IMAGENET_MEAN = IMAGENET_MEAN.to(self.device)
        self.IMAGENET_STD = IMAGENET_STD.to(self.device)
        self.default_prompt = (
            "Hãy trích xuất toàn bộ thông tin từ bức ảnh này theo đúng thứ tự và nội dung như trong ảnh, đảm bảo đầy đủ và chính xác. "
            "Lưu ý: Không thêm bất kỳ bình luận nào khác. Đối với 'Nơi thường trú' và 'Quê quán', hãy trích xuất đầy đủ địa chỉ như trong ảnh, bao gồm cả xã, huyện, tỉnh."
        )
    
    def initialize_model(self):
        """Initialize model with optimized settings"""
        device_map = str(self.device) if self.force_single_gpu else \
                    get_device_map("Vintern-3B-beta", torch.cuda.device_count())
        
        max_memory = {i: "18GB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "30GB"
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
            max_memory=max_memory
        ).eval()
        
        # Store vision device
        self.vision_device = (f"cuda:{device_map.get('vision_model', 0)}" 
                            if isinstance(device_map, dict) and torch.cuda.is_available()
                            else str(self.device))
        
        # Initialize tokenizer with optimized settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            model_max_length=8196
        )
    
    @lru_cache(maxsize=32)
    def get_transform(self, input_size: int) -> T.Compose:
        """Cached transform pipeline"""
        if input_size not in self._transform_cache:
            self._transform_cache[input_size] = T.Compose([
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN.squeeze().tolist(), 
                          std=IMAGENET_STD.squeeze().tolist()),
            ])
        return self._transform_cache[input_size]

    def process_image_chunk(self, img: Image.Image, transform: T.Compose) -> torch.Tensor:
        """Process a single image chunk"""
        return transform(img).to(device=self.vision_device, dtype=torch.bfloat16)

    @lru_cache(maxsize=16)
    def get_target_ratios(self, min_num: int, max_num: int) -> List[Tuple[int, int]]:
        """Cache and generate target ratios for aspect ratio calculation"""
        cache_key = (min_num, max_num)
        if cache_key not in self._target_ratios_cache:
            ratios = set(
                (i, j) 
                for n in range(min_num, max_num + 1) 
                for i in range(1, n + 1) 
                for j in range(1, n + 1) 
                if i * j <= max_num and i * j >= min_num
            )
            self._target_ratios_cache[cache_key] = sorted(ratios, key=lambda x: x[0] * x[1])
        return self._target_ratios_cache[cache_key]

    def find_closest_aspect_ratio(self, 
                                aspect_ratio: float, 
                                target_ratios: List[Tuple[int, int]], 
                                width: int, 
                                height: int, 
                                image_size: int) -> Tuple[int, int]:
        """
        Find the closest aspect ratio match efficiently using vectorized operations
        """
        target_aspect_ratios = np.array([ratio[0] / ratio[1] for ratio in target_ratios])
        ratio_diffs = np.abs(target_aspect_ratios - aspect_ratio)
        
        area = width * height
        area_threshold = 0.5 * image_size * image_size
        
        best_idx = 0
        best_diff = ratio_diffs[0]
        
        for i, diff in enumerate(ratio_diffs):
            ratio = target_ratios[i]
            if diff < best_diff or (diff == best_diff and 
                                  area > area_threshold * ratio[0] * ratio[1]):
                best_diff = diff
                best_idx = i
                
        return target_ratios[best_idx]

    def dynamic_preprocess(self, 
                         image: Image.Image, 
                         min_num: int = 1, 
                         max_num: int = 12, 
                         image_size: int = 448, 
                         use_thumbnail: bool = False) -> List[Image.Image]:
        """
        Optimized dynamic preprocessing with efficient image splitting
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Get cached target ratios
        target_ratios = self.get_target_ratios(min_num, max_num)
        
        # Find optimal aspect ratio
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # Calculate dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize image once
        resized_img = image.resize((target_width, target_height), 
                                 Image.Resampling.LANCZOS)

        # Prepare crop coordinates for all blocks at once
        cols = target_aspect_ratio[0]
        rows = target_aspect_ratio[1]
        
        crop_coords = [
            (
                (i % cols) * image_size,
                (i // cols) * image_size,
                ((i % cols) + 1) * image_size,
                ((i // cols) + 1) * image_size
            )
            for i in range(blocks)
        ]

        # Process crops in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed_images = list(executor.map(
                lambda coords: resized_img.crop(coords), 
                crop_coords
            ))

        # Add thumbnail if needed
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size), 
                                      Image.Resampling.LANCZOS)
            processed_images.append(thumbnail_img)

        return processed_images

    def load_image(self, 
                  image_file: Union[np.ndarray, Image.Image, torch.Tensor],
                  input_size: int = 448,
                  max_num: int = 6) -> torch.Tensor:
        # Convert input to PIL Image
        if isinstance(image_file, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))
        elif isinstance(image_file, Image.Image):
            image = image_file
        elif isinstance(image_file, torch.Tensor):
            image = Image.fromarray(image_file.cpu().numpy().astype(np.uint8))
        else:
            raise ValueError("Unsupported image format")

        # Get cached transform
        transform = self.get_transform(input_size)
        
        # Process image chunks in parallel
        images = self.dynamic_preprocess(image, image_size=input_size,
                                       use_thumbnail=True, max_num=max_num)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_image_chunk, img, transform) 
                      for img in images]
            pixel_values = torch.stack([future.result() for future in futures])
            
        return pixel_values

    @torch.no_grad()
    def process_image(self,
                     image_file: Optional[Union[np.ndarray, Image.Image, torch.Tensor]] = None,
                     custom_prompt: Optional[str] = None,
                     input_size: int = 448,
                     max_num: int = 6) -> str:
        prompt = custom_prompt if custom_prompt else self.default_prompt
        
        if image_file is not None:
            pixel_values = self.load_image(image_file, input_size, max_num)
            question = "<image>\n" + prompt
        else:
            pixel_values = None
            question = prompt

        generation_config = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "num_beams": 2,
            "repetition_penalty": 2.0,
        }

        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
        )
        
        torch.cuda.empty_cache()
        return response