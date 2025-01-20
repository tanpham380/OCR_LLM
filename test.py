import gradio as gr
from controller.openapi_vison import Llm_Vision_Exes
import json
import time

ocr_controller = Llm_Vision_Exes(
    api_key="1234", 
    api_base="http://172.18.249.58:8000/v1",
    generation_config = {
        "best_of": 2
    }
)

def process_image(image):
    try:
        start_time = time.time()
        result = ocr_controller.generate(image)
        content = result.get("content", {})
        if isinstance(content, str):
            content = json.loads(content)
            
        processing_time = round(time.time() - start_time, 2)
        
        response = {
            "result": content,
            "processing_time_s": processing_time
        }
        
        return json.dumps(response, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error processing image: {str(e)}"

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Textbox(label="OCR Result", lines=10),
    title="OCR API Tester",
    description="Upload an image to test the OCR API",
    cache_examples=True
)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")