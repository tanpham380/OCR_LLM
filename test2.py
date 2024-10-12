import cv2
import gradio as gr
import time
from app_utils.image_rectifier_package.image_rectifier import ImageRectify
from controller.llm_vison_future import VinternOCRModel
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import rotate_image
import torch
import numpy as np

# Initialize the LLM Vision model
llm_vison = VinternOCRModel("5CD-AI/Vintern-4B-v1")
idcard_detect = ImageRectify(crop_expansion_factor=0.000001)
orientation_engine = RapidOrientation()

def merge_images_vertically(image1, image2):
    """
    Merge two images vertically with appropriate resizing.
    """
    if image1 is None or image2 is None:
        return None
    # Ensure both images have the same width
    width = max(image1.shape[1], image2.shape[1])
    height1 = int(image1.shape[0] * (width / image1.shape[1]))
    height2 = int(image2.shape[0] * (width / image2.shape[1]))
    resized_image1 = cv2.resize(image1, (width, height1))
    resized_image2 = cv2.resize(image2, (width, height2))

    # Concatenate images vertically
    merged_image = np.vstack((resized_image1, resized_image2))
    return merged_image

def check_and_merge(image1, image2):
    """
    Merge images and display the merged image immediately.
    """
    if image1 is not None and image2 is not None:
        merged_image = merge_images_vertically(image1, image2)
        return merged_image
    else:
        return None

def process_after_merge(merged_image):
    """
    Process the merged image to generate text output and processing time.
    """
    if merged_image is None:
        return None, None

    start_time = time.time()

    # Convert image from RGB to BGR
    image_bgr = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)

    # Detect and rectify the merged image
    detected_image, _ = idcard_detect.detect(image_bgr)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orientation_res, _ = orientation_engine(detected_image)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orientation_res = float(orientation_res)
    if orientation_res != 0:
        detected_image = rotate_image(detected_image, orientation_res)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Process the detected image with LLM
    text_from_vision_model = llm_vison.process_image(detected_image)
    total_time = time.time() - start_time
    processing_time_message = f"Processing Time: {total_time:.2f} seconds"

    return text_from_vision_model, processing_time_message

# Create Gradio interface using Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Vision-based Chat with LLM")
    gr.Markdown("Upload two images, which will be merged and processed by the vision-based LLM to generate text and see the processing time.")

    with gr.Row():
        image_input1 = gr.Image(type="numpy", label="Upload Image 1")
        image_input2 = gr.Image(type="numpy", label="Upload Image 2")

    merged_image_output = gr.Image(type="numpy", label="Merged Image")
    text_output = gr.Textbox(label="Generated Text from Vision Model")
    time_output = gr.Textbox(label="Processing Time")

    image_inputs = [image_input1, image_input2]

    # When images change, merge and display the merged image
    image_input1.change(
        fn=check_and_merge,
        inputs=image_inputs,
        outputs=merged_image_output
    ).then(
        fn=process_after_merge,
        inputs=merged_image_output,
        outputs=[text_output, time_output],
        queue=True
    )

    image_input2.change(
        fn=check_and_merge,
        inputs=image_inputs,
        outputs=merged_image_output
    ).then(
        fn=process_after_merge,
        inputs=merged_image_output,
        outputs=[text_output, time_output],
        queue=True
    )

# Launch the demo with external accessibility
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
