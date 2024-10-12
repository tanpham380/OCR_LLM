import cv2
import gradio as gr
import time  # Import the time module
from app_utils.image_rectifier_package.image_rectifier import ImageRectify
from controller.llm_vison_future import EraxLLMVison , VinternOCRModel
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import rotate_image
import torch

# Initialize the LLM Vision model
llm_vison = VinternOCRModel("5CD-AI/Vintern-4B-v1")
idcard_detect = ImageRectify(crop_expansion_factor = 0.000001)
orientation_engine = RapidOrientation()

def process_image(image):
    start_time = time.time()  # Start timing
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detected_image, _ = idcard_detect.detect(image_bgr)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orientation_res, _ = orientation_engine(detected_image)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orientation_res = float(orientation_res)
    if orientation_res != 0:
        image = rotate_image(detected_image, orientation_res)    
    text_from_vision_model = llm_vison.process_image(image)
    total_time = time.time() - start_time
    print(text_from_vision_model)
    # Format the time into seconds with two decimals
    processing_time_message = f"Processing Time: {total_time:.2f} seconds"

    # Return the detected image, generated text, and processing time
    return image, text_from_vision_model, processing_time_message

# Create Gradio interface
demo = gr.Interface(
    fn=process_image,  # Function to call when image is uploaded
    inputs=gr.Image(type="numpy", label="Upload an image"),  # Image input
    outputs=[
        gr.Image(type="numpy", label="Detected Image"),  # Image output after detection
        gr.Textbox(label="Generated Text from Vision Model"),  # Text output
        gr.Textbox(label="Processing Time")  # Time output
    ],
    title="Vision-based Chat with LLM",  # Title of the demo
    description="Upload an image and interact with the vision-based LLM to generate text and see the processing time."
)

# Launch the demo
if __name__ == "__main__":
    demo.launch()
