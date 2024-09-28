# import gradio as gr
# import torch  # PyTorch for deep learning
# import base64
# import cv2  # OpenCV for image processing
# import time  # For time tracking
# import re  # Regular expressions for date extraction
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info


# class EraXVLPipeline:
#     def __init__(self, model_path):
#         self.model_path = model_path

#         # Load the model with shared memory
#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             self.model_path,
#             torch_dtype=torch.float16,
#             attn_implementation="eager",
#             device_map="auto",
#             offload_state_dict=True
#         )

#         # Load tokenizer and processor
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
#         self.min_pixels = 256 * 28 * 28
#         self.max_pixels = 1280 * 28 * 28
#         self.processor = AutoProcessor.from_pretrained(
#             self.model_path,
#             min_pixels=self.min_pixels,
#             max_pixels=self.max_pixels
#         )

#     def resize_image(self, image, max_size=1024):
#         """Resize the image if larger than max_size."""
#         height, width = image.shape[:2]
#         scaling_factor = max_size / float(max(height, width))
#         if scaling_factor < 1.0:
#             image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))
#         return image

#     def encode_image(self, image, max_size=1024):
#         """Encode image as base64 after resizing."""
#         resized_img = self.resize_image(image, max_size)
#         _, buffer = cv2.imencode('.jpg', resized_img)
#         encoded_image = base64.b64encode(buffer).decode('utf-8')
#         return f"data:image;base64,{encoded_image}"

#     def prepare_message(self, img):
#         """Prepare messages for each image with the text prompt."""
#         base64_data = self.encode_image(img)
#         message = {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": base64_data},
#                 {"type": "text", "text": "Optical Character Recognition for Image"}
#             ]
#         }
#         return message

#     def run_inference_single(self, img):
#         """Run the inference process on a single image."""
#         if img is None:
#             return "No image provided.", 0  # Return error message if image is None
        
#         start_time = time.time()

#         # Prepare the input message
#         message = self.prepare_message(img)

#         # Tokenize text and process the image
#         tokenized_text = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
#         image_inputs, video_inputs = process_vision_info([message])

#         # Prepare the model inputs
#         inputs = self.processor(
#             text=[tokenized_text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt"
#         )
#         inputs = inputs.to("cuda")

#         # Set generation configuration
#         generation_config = self.model.generation_config
#         generation_config.do_sample = True
#         generation_config.temperature = 0.2
#         generation_config.top_k = 1
#         generation_config.top_p = 0.001
#         generation_config.max_new_tokens = 512
#         generation_config.repetition_penalty = 1.1

#         # Generate text based on inputs
#         generated_ids = self.model.generate(**inputs, generation_config=generation_config)

#         # Trim generated tokens and decode the text
#         generated_ids_trimmed = [
#             out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = self.processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )

#         # Measure processing time
#         process_result_time = time.time() - start_time

#         return output_text[0], process_result_time


#     def run_inference(self, img1, img2):
#         """Run the inference process on both images sequentially, and combine results."""
#         # Process the first image
#         result1, time1 = self.run_inference_single(img1)

#         # Process the second image
#         result2, time2 = self.run_inference_single(img2)

#         # Combine the results
#         combined_result = f"Image 1 Result: {result1}\nImage 2 Result: {result2}"
#         total_time = time1 + time2
#         torch.cuda.empty_cache()

#         return combined_result, total_time


# # Helper function to extract dates from text
# def extract_dates(text):
#     # Regular expression to find date-like patterns (dd/mm/yyyy or similar)
#     date_pattern = r"(\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b)"
#     dates = re.findall(date_pattern, text)
#     if dates:
#         return "\n".join(dates)
#     return "No dates found"

# # Create an instance of the EraXVLPipeline
# pipeline = EraXVLPipeline(model_path="erax/EraX-VL-7B-V1")

# # Define Gradio interface function
# def ocr_demo(img1, img2):
#     # Run inference on the images
#     result, duration = pipeline.run_inference(img1, img2)

#     # Extract dates from the combined result
#     dates = extract_dates(result)
#     torch.cuda.empty_cache()
#     # Combine the OCR results with the extracted dates
#     return f"Result:\n{result}\n\nExtracted Dates:\n{dates}\n\nProcessing Time: {duration:.2f} seconds"


# # Define Gradio interface
# iface = gr.Interface(
#     fn=ocr_demo,
#     inputs=[
#         gr.Image(type="numpy", label="Image 1"),
#         gr.Image(type="numpy", label="Image 2")
#     ],
#     outputs="text",
#     title="Optical Character Recognition with Date Extraction",
#     description="Upload two images, and the model will extract text using OCR for each image sequentially. Dates will be extracted from the text automatically.",
#     live=True  # Automatically run when images are uploaded
# )

# # Launch Gradio demo
# if __name__ == "__main__":
#     torch.cuda.empty_cache()  # Clear CUDA cache
#     iface.launch()
