import torch
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import cv2
from PIL import Image
import math

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80, "Vintern-3B-beta": 36
    }[model_name]
    
    # For single GPU, map everything to that device
    if world_size == 1:
        return "cuda:0"
        
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
            
    vision_components = [
        'vision_model',
        'mlp1',
    ]
    
    language_components = [
        'language_model.model.tok_embeddings',
        'language_model.model.embed_tokens',
        'language_model.output',
        'language_model.model.norm',
        'language_model.lm_head',
        'language_model.model.rotary_emb',  
        'language_model.model.wte',         
        'language_model.model.ln_f',        
    ]
    
    for component in vision_components:
        device_map[component] = 0
        
    for component in language_components:
        device_map[component] = 0
        
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    
    return device_map

class VinternOCRModel:
    def __init__(self, 
                model_path="app_utils/weights/Vintern-3B-beta",
                device=None):
        """
        Initialize VinternOCRModel with automatic GPU configuration.
        
        Args:
            model_path (str): Path to the model weights
            device (str or torch.device, optional): Specific device to use. If None, will auto-detect.
                                                Examples: "cuda:0", "cuda:1", "cpu", or torch.device("cuda:0")
        """
        # GPU configuration and device setup
        if device is None:
            # Auto-detect available device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.force_single_gpu = False  # Let it use multiple GPUs if available
        else:
            # Convert string device to torch.device if necessary
            self.device = torch.device(device) if isinstance(device, str) else device
            # Force single GPU if specific device is provided
            self.force_single_gpu = True

        self.model_path = model_path
        
        self.default_prompt = (
            "Hãy trích xuất toàn bộ thông tin từ bức ảnh này theo đúng thứ tự và nội dung như trong ảnh, đảm bảo đầy đủ và chính xác. "
            "Không thêm bất kỳ bình luận nào khác. "
            "Lưu ý: Đối với 'Nơi thường trú' và 'Quê quán', hãy trích xuất đầy đủ địa chỉ như trong ảnh, bao gồm cả xã, huyện, tỉnh.\n"
        )

        # Configure device mapping based on device setting
        if self.force_single_gpu:
            device_map = str(self.device)  # Use the specific device
        else:
            device_map = split_model("Vintern-3B-beta")

        # Load model with appropriate device configuration
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=True,
            device_map=device_map
        ).eval()

        # Store the vision model device for later use
        if isinstance(device_map, dict):
            self.vision_device = f"cuda:{device_map.get('vision_model', 0)}" if torch.cuda.is_available() else "cpu"
        else:
            self.vision_device = device_map if isinstance(device_map, str) else "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=True
        )
        self.tokenizer.model_max_length = 8196

    def build_transform(self, input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
                
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % target_aspect_ratio[0]) * image_size,
                (i // target_aspect_ratio[0]) * image_size,
                ((i % target_aspect_ratio[0]) + 1) * image_size,
                ((i // target_aspect_ratio[0]) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def load_image(self, image_file, input_size=448, max_num=6):
        # Convert input to PIL Image
        if isinstance(image_file, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))
        elif isinstance(image_file, Image.Image):
            image = image_file
        elif isinstance(image_file, torch.Tensor):
            image = Image.fromarray(image_file.cpu().numpy().astype(np.uint8))
        else:
            raise ValueError("Unsupported image format. Provide a NumPy array, PIL Image, or PyTorch tensor.")

        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)

        try:
            pixel_values = torch.stack([transform(img) for img in images])
            # Move pixel_values to the same device as the vision model
            pixel_values = pixel_values.to(device=self.vision_device, dtype=torch.bfloat16)
        except Exception as e:
            raise Exception(f"Error during image transformation: {e}")

        return pixel_values

    def process_image(self, image_file=None, custom_prompt=None, input_size=448, max_num=12):
        prompt = custom_prompt if custom_prompt else self.default_prompt

        if image_file is not None:
            pixel_values = self.load_image(image_file, input_size, max_num)
            question = "<image>\n" + prompt
        else:
            pixel_values = None
            question = prompt

        generation_config = {
            "max_new_tokens": 8196,
            "do_sample": False,
            "num_beams": 2,
            "repetition_penalty": 2.0,
        }

        with torch.no_grad():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
            )

        torch.cuda.empty_cache()
        return response