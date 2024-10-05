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

from qwen_vl_utils import process_vision_info

from app_utils.file_handler import convert_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VinternOCRModel:
    def __init__(self, model_path="5CD-AI/Vintern-1B-v3"):
        """
        Initialize the model and tokenizer.
        """
        self.default_prompt = """trích xuất và trả về văn bản có trong ảnh, không giải thích 
        Lưu ý: lấy cụ thể các thông tin trên thẻ và trả lời dạng Json \n 
        - Chú ý các tường thông tin sau
            + Họ và tên \n
            + Lấy tất cả thông tin Nơi thường trú (place of residence) \n
            + Lấy tất cả thông tin Quê quán ( place of origin , place of birth )\n
            + Ngày tháng năm sinh, ngày tháng năm hết hạn và ngày tháng năm kí của thẻ căn cước \n
        """

        self.model_path = model_path
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
        self.model = torch.compile(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.tokenizer.model_max_length = 4400

    def build_transform(self, input_size):
        """
        Create a transformation pipeline to process input images as np.ndarray.
        """
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def dynamic_preprocess(
        self,
        img: np.ndarray,
        min_num=1,
        max_num=12,
        image_size=448,
        use_thumbnail=False,
    ):
        """
        Dynamically preprocesses the image (np.ndarray) into blocks based on aspect ratio.
        """
        orig_height, orig_width = img.shape[:2]
        aspect_ratio = orig_width / orig_height

        # Calculate existing image aspect ratio and find closest
        target_ratios = [
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        ]
        target_aspect_ratio = min(
            target_ratios, key=lambda x: abs(aspect_ratio - (x[0] / x[1]))
        )

        # Resize the image to fit the aspect ratio
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        img_resized = cv2.resize(img, (target_width, target_height))

        # Split the image into blocks
        block_width = target_width // target_aspect_ratio[0]
        block_height = target_height // target_aspect_ratio[1]
        processed_images = [
            img_resized[y : y + block_height, x : x + block_width]
            for i in range(target_aspect_ratio[0])
            for j in range(target_aspect_ratio[1])
            for x, y in [(i * block_width, j * block_height)]
        ]

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = cv2.resize(img, (image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def load_image(self, image_file: np.ndarray, input_size=448, max_num=30):
        """
        Loads and preprocesses the image (np.ndarray) for OCR.
        Handles block-based dynamic preprocessing and applies necessary transformations.
        """
        transform = self.build_transform(input_size=input_size)

        try:
            # Dynamic preprocessing
            images = self.dynamic_preprocess(
                image_file, image_size=input_size, use_thumbnail=True, max_num=max_num
            )

            # Apply transformations to each block
            pixel_values = torch.stack(
                [transform(img) for img in images]
            )  # Stack into a batch tensor
        except Exception as e:
            raise Exception(f"Error during image transformation: {e}")

        return pixel_values

    def process_image(self, image_file: np.ndarray = None, custom_prompt: str = None):
        """
        Process the input image (np.ndarray) and generate a response from the model.
        Optionally, the image can be omitted by passing None, and a custom prompt can be set.
        """
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self.default_prompt

        if image_file is not None:
            pixel_values = self.load_image(image_file).to(torch.bfloat16).cuda()
            question = "<image>\n" + prompt
        else:
            pixel_values = None  # No image, model may handle this differently
            question = prompt
        generation_config = dict(
            max_new_tokens=2048,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.2,
            temperature=0.2,
            top_p=1.0,
        )

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            history=None,  # Hoặc history=[]
            generation_config=generation_config,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response


# class VinternOCRModel:
#     def __init__(self, model_path="5CD-AI/Vintern-1B-v2"):
#         # Initialization remains the same
#         self.default_prompt = "trích xuất và trả về văn bản có trong ảnh, không thêm bất kỳ mô tả hay giải thích nào."

#         self.model_path = model_path
#         self.model = AutoModel.from_pretrained(
#             model_path,
#             torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,


#         ).eval().cuda()
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
#         self.input_size = 448  # Define input size once

#     def build_transform(self):
#         # Build transform without input_size parameter
#         return T.Compose([
#             T.ToTensor(),
#             T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
#             T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#         ])

#     def load_images(self, image_files: List[np.ndarray]):
#         """
#         Loads and preprocesses a list of images for OCR.
#         """
#         transform = self.build_transform()
#         processed_images = []

#         for img in image_files:
#             # Apply transformations
#             img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             img_transformed = transform(img_pil)
#             processed_images.append(img_transformed)

#         # Stack into a batch tensor
#         pixel_values = torch.stack(processed_images)
#         return pixel_values

#     def process_images(self, image_files: List[np.ndarray], custom_prompt: str = None):
#         """
#         Process a batch of input images and generate responses from the model.
#         """
#         prompt = custom_prompt if custom_prompt else self.default_prompt
#         pixel_values = self.load_images(image_files).to(torch.bfloat16).cuda()
#         batch_size = pixel_values.size(0)

#         # Create batch of prompts
#         questions = [f'<image>\n{prompt}'] * batch_size

#         generation_config = dict(
#     max_new_tokens=500,
#     do_sample=False,
#     num_beams=1,
#     repetition_penalty=1.2,
#     temperature=0.2,
#     top_p=1.0
# )


#         # Process the batch
#         responses = []
#         with torch.no_grad():
#             for i in range(batch_size):
#                 response = self.model.chat(
#                     self.tokenizer,
#                     pixel_values[i].unsqueeze(0),
#                     questions[i],
#                     generation_config
#                 )
#                 responses.append(response)

#         return responses


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
            "text": " trích xuất toàn bộ văn bản của bức ảnh , Không bình luận gì thêm.",
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
        generation_config =  self.model.generation_config
        generation_config.do_sample   = True
        generation_config.temperature = 0.1
        generation_config.top_k       = 1
        generation_config.top_p       = 0.001
        generation_config.max_new_tokens     = 1024
        generation_config.repetition_penalty = 1.1

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
