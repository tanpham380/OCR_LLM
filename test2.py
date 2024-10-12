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

def process_image(image):
    """
    Process a single image: convert color, detect and rectify, correct orientation.
    If the detected side is not the front, crop half of the image.
    Returns the processed image.
    """
    # Convert image from RGB to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect and rectify the image
    detected_image_bgr, is_front = idcard_detect.detect(image_bgr)
    if detected_image_bgr is None:
        return None  # Handle case where detection fails
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # If is_front is False, crop half of the detected image


    # Proceed with orientation correction
    orientation_res, _ = orientation_engine(detected_image_bgr)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orientation_res = float(orientation_res)
    if orientation_res != 0:
        detected_image_bgr = rotate_image(detected_image_bgr, orientation_res)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not is_front:
        # For example, crop the lower half of the image
        height, width = detected_image_bgr.shape[:2]
        cropped_height = height // 2
        detected_image_bgr = detected_image_bgr[:cropped_height, :]
    detected_image_rgb = cv2.cvtColor(detected_image_bgr, cv2.COLOR_BGR2RGB)

    return detected_image_rgb

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

    # Convert merged image to BGR before saving with OpenCV
    merged_image_bgr = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.png", merged_image_bgr)

    return merged_image

def check_and_process(image1, image2):
    """
    Process each image individually, merge them, and then process the merged image with the LLM.
    """
    if image1 is not None and image2 is not None:
        start_time = time.time()

        # Process each image individually
        processed_image1 = process_image(image1)
        processed_image2 = process_image(image2)

        # Merge the processed images
        merged_image = merge_images_vertically(processed_image1, processed_image2)

        # Process the merged image with LLM
        text_from_vision_model = llm_vison.process_image(merged_image)
        total_time = time.time() - start_time
        processing_time_message = f"Processing Time: {total_time:.2f} seconds"

        # Return the merged image, generated text, and processing time
        return merged_image, text_from_vision_model, processing_time_message
    else:
        return None, None, None

# Create Gradio interface using Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Vision-based Chat with LLM")
    gr.Markdown("Upload two images, which will be individually processed, merged, and then passed to the vision-based LLM to generate text and see the processing time.")

    with gr.Row():
        image_input1 = gr.Image(type="numpy", label="Upload Image 1")
        image_input2 = gr.Image(type="numpy", label="Upload Image 2")

    merged_image_output = gr.Image(type="numpy", label="Merged Image")
    text_output = gr.Textbox(label="Generated Text from Vision Model")
    time_output = gr.Textbox(label="Processing Time")

    image_inputs = [image_input1, image_input2]

    # When images change, process and display the outputs
    def update_outputs(image1, image2):
        merged_image, text, time_message = check_and_process(image1, image2)
        return merged_image, text, time_message

    image_input1.change(
        fn=update_outputs,
        inputs=image_inputs,
        outputs=[merged_image_output, text_output, time_output]
    )

    image_input2.change(
        fn=update_outputs,
        inputs=image_inputs,
        outputs=[merged_image_output, text_output, time_output]
    )

# Launch the demo with external accessibility
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
