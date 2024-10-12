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
    
    # Convert BGR to RGB only once if necessary
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect ID card and check if it's the front side
    detected_image, is_front = idcard_detect.detect(image_rgb)
    
    # Process orientation
    orientation_res, _ = orientation_engine(detected_image)
    orientation_res = float(orientation_res)
    
    # Rotate the image only if needed
    if orientation_res != 0:
        detected_image = rotate_image(detected_image, orientation_res)
    
    # Crop the image if it's not the front side
    if not is_front:
        height = detected_image.shape[0] // 2
        detected_image = detected_image[:height, :]

    # Use the LLM vision model to extract text from the image
    text_from_vision_model = llm_vison.process_image(detected_image)
    
    # Calculate total processing time
    total_time = time.time() - start_time
    processing_time_message = f"Processing Time: {total_time:.2f} seconds"
    
    # Empty CUDA cache only if GPU is used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(text_from_vision_model)
    
    return detected_image, text_from_vision_model, processing_time_message


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
    demo.launch(server_name="0.0.0.0")
