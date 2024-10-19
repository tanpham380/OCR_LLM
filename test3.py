import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import gradio as gr
import time
from app_utils.image_rectifier_package.image_rectifier import ImageRectify
from controller.llm_vison_future import VinternOCRModel
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import rotate_image
import logging

logging.basicConfig(level=logging.INFO)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    logging.info(f"Initializing process group for rank {rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    logging.info(f"Process group initialized for rank {rank}")

def cleanup():
    dist.destroy_process_group()

class DistributedVinternOCRModel(VinternOCRModel):
    def __init__(self, model_path, rank, world_size):
        logging.info(f"Initializing DistributedVinternOCRModel for rank {rank}")
        super().__init__(model_path, device=f"cuda:{rank}")
        logging.info(f"Base model initialized for rank {rank}")
        self.model = DDP(self.model, device_ids=[rank])
        logging.info(f"DDP wrapper applied for rank {rank}")
        self.original_chat = self.model.module.chat
        logging.info(f"Original chat method saved for rank {rank}")

    def chat(self, *args, **kwargs):
        return self.original_chat(*args, **kwargs)
    
    
    
    def process_image(self, image_file=None, custom_prompt=None, input_size=448, max_num=12):
        prompt = custom_prompt if custom_prompt else self.default_prompt

        if image_file is not None:
            pixel_values = self.load_image(image_file, input_size, max_num)
            pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
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
            response = self.chat(
                self.tokenizer,
                pixel_values,
                question,
                history=None,
                generation_config=generation_config,
            )

        torch.cuda.empty_cache()

        return response

def run_processing(rank, world_size, model_path):
    logging.info(f"Starting run_processing for rank {rank}")

    setup(rank, world_size)
    
    # Initialize the distributed model
    llm_vision = DistributedVinternOCRModel(model_path, rank, world_size)
    idcard_detect = ImageRectify(crop_expansion_factor=0.0)
    orientation_engine = RapidOrientation()

    def process_image(image):
        start_time = time.time()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detected_image, is_front = idcard_detect.detect(image_bgr)

        torch.cuda.empty_cache()

        orientation_res, _ = orientation_engine(detected_image)
        orientation_res = float(orientation_res)

        if orientation_res != 0:
            detected_image = rotate_image(detected_image, orientation_res)

        if not is_front:
            height = detected_image.shape[0] // 2
            detected_image = detected_image[:height, :]

        torch.cuda.empty_cache()

        image_gray = cv2.cvtColor(detected_image, cv2.COLOR_BGR2GRAY)
        text_from_vision_model = llm_vision.process_image(image_gray)

        detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

        total_time = time.time() - start_time
        processing_time_message = f"Processing time: {total_time:.2f} seconds"

        torch.cuda.empty_cache()

        return detected_image_rgb, text_from_vision_model, processing_time_message

    if rank == 0:
        demo = gr.Interface(
            fn=process_image,
            inputs=gr.Image(type="numpy", label="Upload an image"),
            outputs=[
                gr.Image(type="numpy", label="Processed Image"),
                gr.Textbox(label="Text from Vision Model"),
                gr.Textbox(label="Processing Time")
            ],
            title="Vision-based Chat with LLM",
            description="Upload an image and interact with the Vision-based LLM to generate text and view processing time."
        )
        demo.launch(server_name="0.0.0.0")

    cleanup()

if __name__ == "__main__":
    world_size = 2  # Number of GPUs
    model_path = "app_utils/weights/Vintern-3B-v1-phase4"
    mp.spawn(run_processing, args=(world_size, model_path), nprocs=world_size, join=True)