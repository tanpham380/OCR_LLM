import cv2
import gradio as gr
import time
from app_utils.image_rectifier_package.image_rectifier import ImageRectify
from controller.llm_vison_future import EraxLLMVison, VinternOCRModel
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import rotate_image
import torch

# Initialize the LLM Vision model
llm_vison = VinternOCRModel("5CD-AI/Vintern-4B-v1")
idcard_detect = ImageRectify(crop_expansion_factor=0.000001)
orientation_engine = RapidOrientation()

def process_image(image):
    start_time = time.time()  # Start timing

    # Sử dụng ảnh gốc trong không gian màu BGR
    # Phát hiện ID card và kiểm tra mặt trước
    detected_image, is_front = idcard_detect.detect(image)
    # Giải phóng bộ nhớ CUDA nếu sử dụng GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Xử lý hướng ảnh
    orientation_res, _ = orientation_engine(detected_image)
    orientation_res = float(orientation_res)

    # Xoay ảnh nếu cần thiết
    if orientation_res != 0:
        detected_image = rotate_image(detected_image, orientation_res)

    # Cắt ảnh nếu không phải mặt trước
    if not is_front:
        height = detected_image.shape[0] // 2
        detected_image = detected_image[:height, :]
        # Giải phóng bộ nhớ CUDA nếu sử dụng GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Chuyển đổi ảnh sang RGB trước khi đưa vào mô hình LLM Vision
    detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    text_from_vision_model = llm_vison.process_image(detected_image_rgb)

    # Chuyển đổi ảnh sang RGB trước khi hiển thị trên Gradio
    detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

    # Tính toán thời gian xử lý
    total_time = time.time() - start_time
    processing_time_message = f"Thời gian xử lý: {total_time:.2f} giây"

    # Giải phóng bộ nhớ CUDA nếu sử dụng GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return detected_image_rgb, text_from_vision_model, processing_time_message

# Tạo giao diện Gradio
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Tải lên một ảnh"),
    outputs=[
        gr.Image(type="numpy", label="Ảnh sau khi xử lý"),
        gr.Textbox(label="Văn bản từ mô hình Vision"),
        gr.Textbox(label="Thời gian xử lý")
    ],
    title="Trò chuyện dựa trên Vision với LLM",
    description="Tải lên một ảnh và tương tác với LLM dựa trên Vision để tạo văn bản và xem thời gian xử lý."
)

# Khởi chạy ứng dụng
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
