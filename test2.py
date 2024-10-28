import cv2
import gradio as gr
import time
import torch
import numpy as np
from app_utils.image_rectifier_package.image_rectifier import ImageRectify
from controller.llm_vison_future import VinternOCRModel
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import rotate_image
import threading  # Import threading module

idcard_detect = ImageRectify(crop_expansion_factor=0.02)
orientation_engine = RapidOrientation()
llm_model1 = VinternOCRModel("app_utils/weights/Vintern-3B-beta", device='cuda:0')
llm_model2 = VinternOCRModel("app_utils/weights/Vintern-3B-beta", device='cuda:1')

def process_image(image):
    """
    Process a single image: convert color, detect and rectify, correct orientation.
    If the detected side is not the front, crop half of the image.
    Returns the processed image.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detected_image_bgr, is_front = idcard_detect.detect(image_bgr)
    if detected_image_bgr is None:
        return None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    orientation_res, _ = orientation_engine(detected_image_bgr)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    orientation_res = float(orientation_res)
    if orientation_res != 0:
        detected_image_bgr = rotate_image(detected_image_bgr, orientation_res)

    if not is_front:
        height, width = detected_image_bgr.shape[:2]
        cropped_height = height // 2
        detected_image_bgr = detected_image_bgr[:cropped_height, :]

    detected_image_rgb = cv2.cvtColor(detected_image_bgr, cv2.COLOR_BGR2RGB)
    return detected_image_rgb

def check_and_process(image1, image2):
    """
    Processes the two input images simultaneously, merges them, and returns the 
    merged image and the combined generated text.
    """
    if image1 is None or image2 is None:
        return None, "Please upload two images.", None

    start_time = time.time()

    # Preprocess images
    processed_image1 = process_image(image1)
    processed_image2 = process_image(image2)

    if processed_image1 is None or processed_image2 is None:
        return None, "Image processing failed.", None

    # Initialize response variables
    response1 = None
    response2 = None

    # Define functions to process each image with the corresponding model
    def process_response1():
        nonlocal response1
        response1 = llm_model1.process_image(image_file=processed_image1)

    def process_response2():
        nonlocal response2
        response2 = llm_model2.process_image(image_file=processed_image2)

    # Create threads
    thread1 = threading.Thread(target=process_response1)
    thread2 = threading.Thread(target=process_response2)

    # Start threads
    thread1.start()
    thread2.start()

    # Wait for both threads to complete
    thread1.join()
    thread2.join()

    # Check if both responses were obtained
    if response1 is None or response2 is None:
        return None, "Image processing failed.", None

    # Resize images to have the same height for concatenation
    height1 = processed_image1.shape[0]
    height2 = processed_image2.shape[0]
    if height1 != height2:
        # Adjust widths proportionally
        if height1 > height2:
            new_width = int(processed_image2.shape[1] * (height1 / height2))
            processed_image2 = cv2.resize(processed_image2, (new_width, height1))
        else:
            new_width = int(processed_image1.shape[1] * (height2 / height1))
            processed_image1 = cv2.resize(processed_image1, (new_width, height2))

    # Concatenate images horizontally
    merged_image = np.concatenate((processed_image1, processed_image2), axis=1)

    end_time = time.time()
    processing_time = end_time - start_time

    time_message = f"Processing time: {processing_time:.2f} seconds"
    return merged_image, response1 + "\n" + response2, time_message


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
