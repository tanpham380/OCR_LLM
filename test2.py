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

def process_images(image1, image2):
    if image1 is None or image2 is None:
        return None, None, None  # Return None if images are not provided

    start_time = time.time()  # Start timing

    # Convert images from RGB to BGR
    image1_bgr = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    image2_bgr = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    # Detect and rectify the first image
    detected_image1, _ = idcard_detect.detect(image1_bgr)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orientation_res1, _ = orientation_engine(detected_image1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orientation_res1 = float(orientation_res1)
    if orientation_res1 != 0:
        detected_image1 = rotate_image(detected_image1, orientation_res1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Detect and rectify the second image
    detected_image2, _ = idcard_detect.detect(image2_bgr)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orientation_res2, _ = orientation_engine(detected_image2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orientation_res2 = float(orientation_res2)
    if orientation_res2 != 0:
        detected_image2 = rotate_image(detected_image2, orientation_res2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Merge the two images vertically
    merged_image = merge_images_vertically(detected_image1, detected_image2)

    # Process the merged image with LLM
    text_from_vision_model = llm_vison.process_image(merged_image)

    total_time = time.time() - start_time
    print(text_from_vision_model)
    # Format the time into seconds with two decimals
    processing_time_message = f"Processing Time: {total_time:.2f} seconds"

    # Return the merged image, generated text, and processing time
    return merged_image, text_from_vision_model, processing_time_message

def merge_images_vertically(image1, image2):
    """
    Merge two images vertically with appropriate resizing.
    """
    # Ensure both images have the same width
    width = max(image1.shape[1], image2.shape[1])
    height1 = int(image1.shape[0] * (width / image1.shape[1]))
    height2 = int(image2.shape[0] * (width / image2.shape[1]))
    resized_image1 = cv2.resize(image1, (width, height1))
    resized_image2 = cv2.resize(image2, (width, height2))

    # Concatenate images vertically
    merged_image = np.vstack((resized_image1, resized_image2))
    # Optionally save the merged image for debugging
    cv2.imwrite("test.png", merged_image)
    return merged_image

# Create Gradio interface using Blocks for better control
with gr.Blocks() as demo:
    gr.Markdown("# Vision-based Chat with LLM")
    gr.Markdown("Upload two images, which will be merged and processed by the vision-based LLM to generate text and see the processing time.")

    with gr.Row():
        image_input1 = gr.Image(type="numpy", label="Upload Image 1")
        image_input2 = gr.Image(type="numpy", label="Upload Image 2")

    process_button = gr.Button("Process Images")

    # Outputs are hidden initially
    merged_image_output = gr.Image(type="numpy", label="Merged Image", visible=False)
    text_output = gr.Textbox(label="Generated Text from Vision Model", visible=False)
    time_output = gr.Textbox(label="Processing Time", visible=False)

    # Define the function to update the visibility of outputs
    def update_visibility(merged_image, text, time_message):
        if merged_image is None:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    # When the button is clicked, process the images and update outputs
    process_button.click(
        fn=process_images,
        inputs=[image_input1, image_input2],
        outputs=[merged_image_output, text_output, time_output]
    ).then(
        fn=update_visibility,
        inputs=[merged_image_output, text_output, time_output],
        outputs=[merged_image_output, text_output, time_output]
    )

# Launch the demo with external accessibility
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
