import cv2
from controller.llm_vison_future import EraxLLMVison

# Initialize the LLM Vision model
llm_vison = EraxLLMVison("app_utils/weights/EraX-VL-7B-V1")

# Load the image
image_path = "/home/gitlab/ocr/2024_01_22_10_28_55_resize.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (1048, 1048))

# Set prompt messages with the image
messages = llm_vison.set_prompt_messages(image)

# Generate response from LLM Vision (chat with the image and prompt)
text_from_vision_model = llm_vison.chat(messages)

# Print the object and the generated text
print(f"Generated text from vision model:\n{text_from_vision_model}")
