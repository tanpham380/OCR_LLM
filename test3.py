import requests
import json
import base64
import sys
from typing import Optional

def check_ollama_status(base_url: str) -> bool:
    """Check if Ollama server is running"""
    try:
        response = requests.get(f"{base_url}/api/tags")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def chat_with_image(image_path: str, 
                   prompt: str, 
                   ollama_base_url: str = "http://localhost:11434",
                   model: str = "llama2",
                   timeout: int = 30) -> Optional[str]:
    """
    Sends a chat request to Ollama with an image and a prompt.
    """
    if not check_ollama_status(ollama_base_url):
        raise ConnectionError("Ollama server is not running")

    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        raise IOError(f"Failed to read image file: {e}")

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False
    }

    try:
        response = requests.post(
            f"{ollama_base_url}/api/generate", 
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        if response.status_code == 500:
            error_content = response.text
            raise ValueError(f"Server error: {error_content}")
            
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"API request failed: {e}")
    except (KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid response format: {e}")

if __name__ == "__main__":
    image_path = "/home/gitlab/ocr/test.jpg"
    prompt = "Diễn tả nội dung bức ảnh này"
    model = "hf.co/mradermacher/EraX-VL-2B-V1.5-GGUF:Q8_0"
    ollama_url = "http://localhost:11434"

    try:
        response = chat_with_image(
            image_path=image_path, 
            prompt=prompt, 
            ollama_base_url=ollama_url,
            model=model,
            timeout=60
        )
        print(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)