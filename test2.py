import os
from app_utils.prompt import CCCD_BACK_PROMPT, CCCD_FRONT_PROMPT
import gradio as gr
from controller.vllm_qwen import VLLM_Exes


# Initialize LLM controller
vison_model = VLLM_Exes()


examples = [
    # Front ID examples
    [None, CCCD_FRONT_PROMPT],
    # Back ID examples
    [None, CCCD_BACK_PROMPT]
]
async def process_image_async(image_path, question):
    try:
        if not image_path or not question:
            raise ValueError("Image and question are required")
            
        if not os.path.exists(image_path):
            raise ValueError("Image file not found")
            

            
        response =  vison_model.generate(
            prompt=question, 
            image_file=image_path,
            # system_prompt=DEAFULT_SYSTEM_PROMPT,
        )
        
        if not response:
            raise ValueError("Empty response from model")
            
        return response
        
    except Exception as e:
        print(f"Error details: {str(e)}")  # Debug log
        return f"Error processing image: {str(e)}"
demo = gr.Interface(
    fn=process_image_async,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Textbox(label="Ask a question about the image")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Visual Question Answering Demo",
    description="Upload an image and ask a question about it",
    examples=examples
)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )