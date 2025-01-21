import gradio as gr
from app_utils.prompt import VINTERN_CCCD_BACK_PROMPT, VINTERN_CCCD_FRONT_PROMPT
from controller.openapi_vison import Llm_Vision_Exes
import json
import asyncio
import time
import re

def parse_llm_response(content_str: str):
    """Enhanced JSON parser for LLM responses"""
    if not content_str or not isinstance(content_str, str):
        return {"error": "No content to parse."}
        
    def try_parse(s):
        try:
            return json.loads(s)
        except:
            return None
            
    try:
        # Debug original content
        print(f"Original content: {content_str}")
        
        # Basic cleaning
        content_str = content_str.strip()
        
        # Try parsing original content first
        result = try_parse(content_str)
        if result:
            return result
            
        # Handle markdown blocks
        if '```' in content_str:
            blocks = [b.strip() for b in content_str.split('```')]
            for block in blocks:
                if '{' in block and '}' in block:
                    result = try_parse(block)
                    if result:
                        return result
                        
        # Find and parse JSON object
        start = content_str.find('{')
        end = content_str.rfind('}')
        if start != -1 and end != -1:
            json_str = content_str[start:end + 1]
            result = try_parse(json_str)
            if result:
                return result
                
        # Final cleanup attempt
        cleaned = content_str.replace('\n', ' ').replace('\r', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.replace("'", '"')
        
        result = try_parse(cleaned)
        if result:
            return result
            
        return {"error": "Failed to parse JSON after all attempts"}
        
    except Exception as e:
        print(f"Parse error: {str(e)}")
        return {"error": f"Parsing error: {str(e)}"}

ocr_controller = Llm_Vision_Exes(
    api_key="1234", 
    api_base="http://172.18.249.58:8000/v1",
    generation_config = {
        "best_of": 1
    }
)

ocr_controller2 = Llm_Vision_Exes(
    api_key="1234", 
    api_base="http://172.18.249.58:8001/v1",
    generation_config = {
        "best_of": 1
    }
)

async def process_front_image(image):
    try:
        result = ocr_controller.generate(image, VINTERN_CCCD_FRONT_PROMPT)
        content_str = result.get("content", "")
        front_dict = parse_llm_response(content_str)
        return front_dict
    except Exception as e:
        return {"error": str(e)}

async def process_back_image(image):
    try:
        result = ocr_controller2.generate(image, VINTERN_CCCD_BACK_PROMPT)
        content_str = result.get("content", "")
        back_dict = parse_llm_response(content_str)
        return back_dict
    except Exception as e:
        return {"error": str(e)}

async def process_images(front_image, back_image):
    start_time = time.time()
    
    # Gọi 2 hàm xử lý LLM cho mặt trước và mặt sau đồng thời
    front_result, back_result = await asyncio.gather(
        process_front_image(front_image),
        process_back_image(back_image)
    )
    
    processing_time = round(time.time() - start_time, 2)
    print(f"Processing time: {processing_time}s")

    # Ta sẽ gộp tất cả các trường (key) của front_result và back_result
    # vào chung 1 dict "information"
    combined_information = {}
    # Nếu front_result là dict, update vào combined_information
    if isinstance(front_result, dict):
        combined_information.update(front_result)
    # Nếu back_result là dict, update tiếp vào combined_information
    if isinstance(back_result, dict):
        combined_information.update(back_result)

    # Tạo JSON cuối cùng
    combined_result = {
        "information": combined_information,
        "processing_time_s": processing_time
    }

    # Trả về JSON dạng chuỗi
    return json.dumps(combined_result, indent=2, ensure_ascii=False)

# Tạo giao diện Gradio
demo = gr.Interface(
    fn=process_images,
    inputs=[
        gr.Image(type="filepath", label="Upload Front Side"),
        gr.Image(type="filepath", label="Upload Back Side")
    ],
    outputs=gr.Textbox(label="OCR Result", lines=15),
    title="CCCD OCR API Tester",
    description="Upload front and back images of CCCD to test the OCR API",
    cache_examples=True
)

if __name__ == "__main__":
    demo.queue().launch(share=False, server_name="0.0.0.0")
