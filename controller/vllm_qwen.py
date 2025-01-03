import asyncio
import functools
import os
from typing import Optional, Union, Dict, Any, List

import openai
from openai import OpenAI
import aiohttp
import base64
import io
import cv2
import numpy as np
import torch
from PIL import Image

class VLLM_Exes:
    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model_name: str = "erax-ai/EraX-VL-2B-V1.5",
        api_key: Optional[str] = None,
        **kwargs
    ):
        self.api_base = api_base
        self.model_name = model_name
        self.api_key = api_key or os.getenv("API_KEY") or "EMPTY"
        self.session = None
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )

    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def _prepare_image_input(
            self,
            image_file: Union[str, np.ndarray, Image.Image, torch.Tensor]
        ) -> str:
        try:
            if isinstance(image_file, str) and image_file.startswith("data:"):
                return image_file

            # Get initial image
            if isinstance(image_file, str):
                if image_file.startswith("http"):
                    await self._ensure_session()
                    timeout = aiohttp.ClientTimeout(total=30)
                    retry_count = 3
                    last_error = None
                    
                    for attempt in range(retry_count):
                        try:
                            async with self.session.get(image_file, timeout=timeout) as response:
                                response.raise_for_status()
                                img_bytes = await response.read()
                                img = Image.open(io.BytesIO(img_bytes))
                                break
                        except Exception as e:
                            last_error = e
                            if attempt < retry_count - 1:
                                await asyncio.sleep(1 * (attempt + 1))
                            else:
                                raise ValueError(f"Failed to download image after {retry_count} attempts: {str(last_error)}")
                else:
                    img = Image.open(image_file)
            elif isinstance(image_file, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))
            elif isinstance(image_file, torch.Tensor):
                arr = image_file.cpu().numpy().astype(np.uint8)
                img = Image.fromarray(arr)
            elif isinstance(image_file, Image.Image):
                img = image_file
            else:
                raise ValueError(f"Unsupported image type: {type(image_file)}")

            # Resize if image is too large
            max_size = 800
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Compress image more aggressively for large images
            quality = 85 if max(img.size) <= 400 else 65
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{encoded}"

        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    async def generate(
        self,
        prompt: Union[str, Dict],
        image_file: Optional[str] = None,
        system_prompt: str = "You are a helpful AI assistant that describes images accurately and in detail.",
        custom_sampling_params: Optional[Dict[str, Any]] = None
    ) -> str:
        await self._ensure_session()
        custom_sampling_params = custom_sampling_params or {}

        base_prompt = prompt if isinstance(prompt, str) else str(prompt)
        temperature = custom_sampling_params.get("temperature", 0.7)
        max_tokens = min(
            custom_sampling_params.get("max_tokens", 1024),
            self.max_tokens - 2048
        )
        timeout = custom_sampling_params.get("timeout", 60)

        try:
            print(f"Preparing request with prompt: {base_prompt[:100]}...")
            
            content_user = base_prompt
            if image_file:
                image_data = await self._prepare_image_input(image_file)
                print(f"Image data prepared, length: {len(image_data)}")
                content_user = [
                    {"type": "text", "text": base_prompt},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]

            print(f"Sending request to model: {self.model_name}")
            print(f"Max tokens for completion: {max_tokens}")
            
            def async_create():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content_user}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )

            response = async_create()

            print(f"Response status: {response.choices[0].finish_reason}")
            print(f"Tokens used: {response.usage.total_tokens}")
            
            if not response.choices or not response.choices[0].message.content:
                raise ValueError(f"No valid response generated. Finish reason: {response.choices[0].finish_reason}")

            return response.choices[0].message.content

        except openai.Timeout as e:
            print(f"OpenAI API request timed out: {e}")
            raise TimeoutError(f"Request timed out after {timeout} seconds")
        except Exception as e:
            print(f"[Detailed Error] {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response status: {getattr(e.response, 'status', 'unknown')}")
                print(f"Response text: {getattr(e.response, 'text', 'unknown')}")
            raise


    def generate_sync(
        self,
        prompt: Union[str, Dict],
        image_file: Optional[str] = None,
        system_prompt: Optional[str] = None,
        custom_sampling_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Synchronous wrapper for generate()"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.generate(prompt, image_file, system_prompt, custom_sampling_params)
        )

    async def generate_batch(
        self,
        prompts: List[Union[str, Dict]],
        image_files: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        custom_sampling_params: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 5
    ) -> List[str]:
        """Process multiple prompts concurrently"""
        await self._ensure_session()

        if image_files and len(image_files) != len(prompts):
            raise ValueError("Number of images must match number of prompts")

        image_files = image_files or [None] * len(prompts)

        sem = asyncio.Semaphore(max_concurrent)

        async def worker(p, img):
            async with sem:
                return await self.generate(
                    p, 
                    image_file=img,
                    system_prompt=system_prompt,
                    custom_sampling_params=custom_sampling_params
                )

        tasks = [worker(p, img) for p, img in zip(prompts, image_files)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = []
        for res in results:
            if isinstance(res, Exception):
                print(f"Batch item failed: {str(res)}")
                processed_results.append(None)
            else:
                processed_results.append(res)

        return processed_results

    def generate_batch_sync(
        self,
        prompts: List[Union[str, Dict]],
        image_files: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        custom_sampling_params: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 5
    ) -> List[str]:
        """Synchronous wrapper for generate_batch"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.generate_batch(
                prompts,
                image_files,
                system_prompt,
                custom_sampling_params,
                max_concurrent
            )
        )

    @classmethod
    def create_with_retry(
        cls,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> "VLLM_Exes":
        """Create instance with retry mechanism"""
        instance = cls(**kwargs)

        original_generate = instance.generate

        @functools.wraps(original_generate)
        async def retry_wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await original_generate(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
            raise last_error

        instance.generate = retry_wrapper
        return instance

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    def __enter__(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._ensure_session())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.__aexit__(exc_type, exc_val, exc_tb))

if __name__ == "__main__":
    async def test_code():
        # Create instance with retry mechanism
        vllm = VLLM_Exes.create_with_retry(
            api_base="http://localhost:8000/v1",
            model_name="erax-ai/EraX-VL-2B-V1.5",
            max_retries=3
        )
        
        try:
            async with vllm:
                # Test 1: Simple text generation
                print("\nTesting text generation...")
                text_result = await vllm.generate(
                    prompt="What is 2+2? Explain step by step.",
                    custom_sampling_params={"max_tokens": 512}
                )
                print("[Text result]:", text_result)
                
                # Test 2: Single image processing
                print("\nTesting image generation...")
                image_result = await vllm.generate(
                    prompt="What do you see in this image? Please describe it in detail.",
                    image_file="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    custom_sampling_params={"max_tokens": 512}
                )
                print("[Image result]:", image_result)
                
        except Exception as e:
            print(f"Test failed with error: {str(e)}")
            raise

    # Run the test
    asyncio.run(test_code())