import base64
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import cv2
from PIL import Image

from qwen_vl_utils import process_vision_info

from app_utils.file_handler import convert_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# class VinternOCRModel:
#     def __init__(self, model_path="5CD-AI/Vintern-1B-v3"):
#         """
#         Initialize the model and tokenizer.
#         """
#         self.default_prompt = (
#             "Hãy trích xuất toàn bộ chi tiết của bức ảnh này theo đúng thứ tự của nội dung và đầy đủ trong ảnh, đảm bảo đúng chính tả Không bình luận gì thêm.  "
#             "Chỉ trả lại bằng tiếng Việt.\n"
#             "Lưu ý: lấy cụ thể các thông tin trên thẻ một cách chính xác theo từng trường:\n"
#         )
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.model_path = model_path
#         self.model = AutoModel.from_pretrained(
#                     model_path,
#                     torch_dtype=torch.bfloat16,
#                     low_cpu_mem_usage=True,
#                     trust_remote_code=True,
#                 ).to(self.device)
#         self.model.eval()
#         # self.model = torch.compile(self.model)

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_path, trust_remote_code=True, use_fast=False
#         )
#         self.tokenizer.model_max_length = 4400

#     def build_transform(self, input_size):
#         """
#         Create a transformation pipeline to process input images as np.ndarray.
#         """
#         return T.Compose(
#             [
#                 T.ToTensor(),
#                 T.Resize(
#                     (input_size, input_size), interpolation=InterpolationMode.BICUBIC
#                 ),
#                 T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#             ]
#         )

#     def dynamic_preprocess(
#         self,
#         img: np.ndarray,
#         min_num=1,
#         max_num=12,
#         image_size=448,
#         use_thumbnail=False,
#     ):
#         """
#         Dynamically preprocesses the image (np.ndarray) into blocks based on aspect ratio.
#         """
#         orig_height, orig_width = img.shape[:2]
#         aspect_ratio = orig_width / orig_height

#         # Calculate existing image aspect ratio and find closest
#         target_ratios = [
#             (rows, cols)
#             for n in range(min_num, max_num + 1)
#             for rows in range(1, n + 1)
#             for cols in range(1, n + 1)
#             if rows * cols == n
#         ]
#         target_aspect_ratio = min(
#             target_ratios, key=lambda x: abs(aspect_ratio - (x[0] / x[1]))
#         )

#         # Resize the image to fit the aspect ratio
#         target_width = image_size * target_aspect_ratio[0]
#         target_height = image_size * target_aspect_ratio[1]
#         img_resized = cv2.resize(img, (target_width, target_height))

#         # Split the image into blocks
#         block_width = target_width // target_aspect_ratio[0]
#         block_height = target_height // target_aspect_ratio[1]
#         processed_images = [
#             img_resized[y : y + block_height, x : x + block_width]
#             for i in range(target_aspect_ratio[0])
#             for j in range(target_aspect_ratio[1])
#             for x, y in [(i * block_width, j * block_height)]
#         ]

#         if use_thumbnail and len(processed_images) != 1:
#             thumbnail_img = cv2.resize(img, (image_size, image_size))
#             processed_images.append(thumbnail_img)

#         return processed_images

#     def load_image(self, image_file: np.ndarray, input_size=448, max_num=12):
#         """
#         Loads and preprocesses the image (np.ndarray) for OCR.
#         Handles block-based dynamic preprocessing and applies necessary transformations.
#         """
#         transform = self.build_transform(input_size=input_size)

#         try:
#             # Dynamic preprocessing
#             images = self.dynamic_preprocess(
#                 image_file, image_size=input_size, use_thumbnail=True, max_num=max_num
#             )

#             # Apply transformations to each block
#             pixel_values = torch.stack(
#                 [transform(img) for img in images]
#             )  # Stack into a batch tensor
#         except Exception as e:
#             raise Exception(f"Error during image transformation: {e}")

#         return pixel_values

#     def process_image(self, image_file: np.ndarray = None, custom_prompt: str = None , input_size : int  = 448 , max_num: int  = 30):
#         """
#         Process the input image (np.ndarray) and generate a response from the model.
#         Optionally, the image can be omitted by passing None, and a custom prompt can be set.
#         """
#         if custom_prompt:
#             prompt = custom_prompt
#         else:
#             prompt = self.default_prompt

#         if image_file is not None:
#             pixel_values = self.load_image(image_file , input_size , max_num).to(torch.bfloat16).cuda()
#             question = "<image>\n" + prompt
#         else:
#             pixel_values = None  # No image, model may handle this differently
#             question = prompt
#         generation_config = dict(
#             max_new_tokens=8192,
#             do_sample=True,
#             num_beams=2,
#             repetition_penalty=1.2,
#             temperature=0.2,
#             top_p=1.0,
#         )
#         # generation_config = dict(max_new_tokens= 64, do_sample=False, num_beams = 1, repetition_penalty=3.5)

#         with torch.no_grad():
#             response = self.model.chat(
#                     self.tokenizer,
#                     pixel_values,
#                     question,
#                     history=None,  # Hoặc history=[]
#                     generation_config=generation_config,
#                 )

#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         return response


class VinternOCRModel:
    def __init__(self, model_path="5CD-AI/Vintern-1B-v3"):
        """
        Initialize the model and tokenizer.
        """
        self.default_prompt = (
            "Hãy trích xuất toàn bộ chi tiết của bức ảnh này theo đúng thứ tự của nội dung và đầy đủ trong ảnh, "
            "đảm bảo đúng chính tả. Không bình luận gì thêm. "
            "Chỉ trả lại bằng tiếng Việt.\n"
            "Lưu ý: lấy cụ thể các thông tin trên thẻ một cách chính xác theo từng trường:\n"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = model_path
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.tokenizer.model_max_length = 4400

    def build_transform(self, input_size):
        """
        Create a transformation pipeline to process input images as PIL Images.
        """
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """
        Find the closest aspect ratio from target ratios.
        """
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """
        Dynamically preprocess the image into blocks based on aspect ratio.
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate possible grid sizes
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height))

        # Split the image into blocks
        processed_images = []
        for i in range(blocks):
            box = (
                (i % target_aspect_ratio[0]) * image_size,
                (i // target_aspect_ratio[0]) * image_size,
                ((i % target_aspect_ratio[0]) + 1) * image_size,
                ((i // target_aspect_ratio[0]) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def load_image(self, image_file, input_size=448, max_num=30):
        """
        Loads and preprocesses the image for OCR.
        Handles block-based dynamic preprocessing and applies necessary transformations.
        """
        if isinstance(image_file, np.ndarray):
            # Convert NumPy array to PIL Image
            image = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))
        elif isinstance(image_file, Image.Image):
            image = image_file
        else:
            raise ValueError("Unsupported image format. Provide a NumPy array or PIL Image.")

        transform = self.build_transform(input_size=input_size)

        # Dynamic preprocessing
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )

        # Apply transformations to each block
        try:
            pixel_values = [transform(img) for img in images]
            pixel_values = torch.stack(pixel_values)
        except Exception as e:
            raise Exception(f"Error during image transformation: {e}")

        return pixel_values

    def process_image(self, image_file=None, custom_prompt=None, input_size=448, max_num=30):
        """
        Process the input image and generate a response from the model.
        Optionally, the image can be omitted by passing None, and a custom prompt can be set.
        """
        prompt = custom_prompt if custom_prompt else self.default_prompt

        if image_file is not None:
            pixel_values = self.load_image(image_file, input_size, max_num)
            pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
            question = "<image>\n" + prompt
        else:
            pixel_values = None  # No image provided
            question = prompt

        generation_config = {
            "max_new_tokens": 4096,  # Adjust as needed
            "do_sample": False,
            "num_beams": 2,
            "repetition_penalty": 1.2,
            "temperature": 0.2,
            "top_p": 1.0,
        }

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                history=None,
                generation_config=generation_config,
            )

        # Clear GPU cache
        torch.cuda.empty_cache()

        return response


class EraxLLMVison:
    def __init__(self, model_path: str = "erax/EraX-VL-7B-V1") -> None:
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # replace with "flash_attention_2" if your GPU is Ampere architecture
            device_map="auto",
            max_memory={0: "23GiB", "cpu": "30GiB"},  # Limit GPU memory usage

        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels = 1024 * 28 * 28,

            max_pixels=1280 * 28 * 28,
        )

    def __load_image(self, image: np.ndarray) -> str:
        # image = cv2.resize(image, (720, 720))
        image_bytes = convert_image(image)
        encoded_image = base64.b64encode(image_bytes)
        decoded_image_text = encoded_image.decode("utf-8")
        return f"data:image;base64,{decoded_image_text}"

    def set_prompt_messages(
        self,
        image: np.ndarray = None,
        custom_user_prompt: str = "",
        custom_system_prompt: str = "",
    ):

        # Set default prompts if not provided
        default_user_prompt = {
            "type": "text",
            "text": " Hãy trích xuất toàn bộ chi tiết của bức ảnh này theo đúng thứ tự của nội dung và đầy đủ trong ảnh Không bình luận gì thêm. Lưu ý: trích xuất ra hết theo thứ tự và định dạng của câu chữ và đảm bảo đúng chính tả, Chỉ trả lại bằng tiếng Việt.",
        }
        if custom_system_prompt:
            system_text = custom_system_prompt
        else:
            system_text = ("Trả lời dạng Json. ")
        
        default_system_prompt = {
            "role": "system",
            "content": system_text
            
        }

        # Use default prompts if custom ones are not provided
        user_prompt = custom_user_prompt or default_user_prompt

        try:
            messages = [default_system_prompt]

            # Check if an image is provided
            if image is not None:
                image_base64_data = self.__load_image(image)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_base64_data},
                            user_prompt,
                        ],
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                )

            return messages
        except Exception as e:
            raise Exception(f"Error while setting prompt messages: {str(e)}")

    def chat(self,messages: str ) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = inputs.to("cuda")
        inputs = inputs.to('cuda', non_blocking=True)
        generation_config = self.model.generation_config
        generation_config.do_sample = True
        generation_config.temperature = 0.1 # Corrected from (0.1111,) to 0.1111 (float)
        generation_config.top_k = 10  # Ensure this is an int, not a tuple
        generation_config.top_p = 0.1  # Ensure this is a float, not a tuple
        generation_config.max_new_tokens = 8192
        generation_config.repetition_penalty = 1.1

        # generation_config =  self.model.generation_config
        # generation_config.do_sample   = True
        # generation_config.temperature = 0.1111,
        # generation_config.top_k       =  int(10)
        # generation_config.top_p       = float(0.1)
        # generation_config.max_new_tokens     = 8192
        # generation_config.repetition_penalty = float(1.1)

        generated_ids = self.model.generate(**inputs, generation_config=generation_config)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
