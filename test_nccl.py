from aphrodite import EngineArgs, LLM
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_engine_args() -> EngineArgs:
    """Initialize engine arguments"""
    return EngineArgs(
        model="erax-ai/EraX-VL-2B-V1.5",
        max_model_len=2048,
        rope_scaling={
            "type": "dynamic",
            "factor": 2.0,
        },
        dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=1,
        disable_custom_all_reduce=True
    )

def main():
    try:
        # Initialize args and LLM with model name
        args = init_engine_args()
        model_name = "erax-ai/EraX-VL-2B-V1.5"
        llm = LLM(model=model_name, engine_args=args)
        
        # Process image
        image_path = '/home/gitlab/ocr/test.png'
        image = Image.open(image_path)
        
        # Generate response
        prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"
        outputs = llm.generate({
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        })
        
        # Print results
        for output in outputs:
            print(output.outputs[0].text)
            
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main()