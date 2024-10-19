import cv2
import gradio as gr
import time
import torch
import numpy as np
from multiprocessing import Process, Queue, set_start_method
from app_utils.image_rectifier_package.image_rectifier import ImageRectify
from controller.llm_vison_future import VinternOCRModel
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import rotate_image

# Set the multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == "__main__":
    set_start_method('spawn')

idcard_detect = ImageRectify(crop_expansion_factor=0.02)
orientation_engine = RapidOrientation()

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

def merge_images_vertically(image1, image2):
    """
    Merge two images vertically with appropriate resizing.
    """
    if image1 is None or image2 is None:
        return None
    width = max(image1.shape[1], image2.shape[1])
    height1 = int(image1.shape[0] * (width / image1.shape[1]))
    height2 = int(image2.shape[0] * (width / image2.shape[1]))
    resized_image1 = cv2.resize(image1, (width, height1))
    resized_image2 = cv2.resize(image2, (width, height2))
    merged_image = np.vstack((resized_image1, resized_image2))
    merged_image_bgr = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.png", merged_image_bgr)
    return merged_image

def process_in_separate_gpu(image, model_path, device, queue):
    """
    Load the model within the process and process the image.
    """
    # Load the model inside the process to avoid pickling issues
    llm_model = VinternOCRModel(model_path, device=device)
    result = llm_model.process_image(image)
    queue.put(result)

def check_and_process(image1, image2):
    """
    Process each image individually, merge them, and then process the merged image with the LLM in separate processes.
    """
    if image1 is not None and image2 is not None:
        start_time = time.time()

        # Process each image individually
        processed_image1 = process_image(image1)
        processed_image2 = process_image(image2)

        # Merge the processed images
        merged_image = merge_images_vertically(processed_image1, processed_image2)

        # Create queues to capture results
        queue1 = Queue()
        queue2 = Queue()

        # Paths to the model weights
        model_path = "app_utils/weights/Vintern-4B-v1"

        # Create separate processes for each model
        process1 = Process(target=process_in_separate_gpu, args=(processed_image1, model_path, 'cuda:0', queue1))
        process2 = Process(target=process_in_separate_gpu, args=(processed_image2, model_path, 'cuda:1', queue2))

        # Start the processes
        process1.start()
        process2.start()

        # Wait for the processes to finish
        process1.join()
        process2.join()

        # Get the results from the queues
        result1 = queue1.get()
        result2 = queue2.get()

        # Combine the results from both GPUs
        text_from_vision_model = " ".join([result1, result2])
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
