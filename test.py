import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

torch.set_default_device('cuda')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

model_name = f"/home/gitlab/ocr/pretrained/Vintern_v2_rotate"

model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

def chat(message, state=None):
    test_image = message["files"][0]["path"]
    pixel_values = load_image(test_image, max_num=12).to(torch.bfloat16).cuda()

    # System Prompt: Instruct the model on how to respond
    # system_prompt = "Chỉ trả lời với một con số biểu thị góc xoay cần thiết (theo độ) để căn cước công dân trong ảnh được xoay đúng chiều, với góc 0 độ là văn bản nằm ngang và đọc được từ trái sang phải."

    # User Prompt: This is the user's input prompt for the task
    user_prompt = message["text"]

    # Combine system prompt and user prompt
    question = "\n" + '<image>\n' + user_prompt

    # Adjust generation configuration for short numerical response
    generation_config = dict(max_new_tokens=10, do_sample=False, num_beams=1, repetition_penalty=2.5)

    # Generate response
    response, conv_history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    
    print(f'User: {question}\nAssistant: {response}')

    return response

CSS = """
/* Your CSS code */
"""

demo = gr.ChatInterface(
    fn=chat,
    description="""Try [Vintern-1B-v3](https://huggingface.co/5CD-AI/Vintern-1B-v3) in this demo. Vintern-1B-v2 consists of [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px), an MLP projector, and [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct).""",
    title="❄️ Vintern-1B-v3 ❄️",
    multimodal=True,
    css=CSS
)
demo.queue().launch()
