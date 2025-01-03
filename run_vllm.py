#!/usr/bin/env python3
import os
import subprocess
import yaml
import json
import nvidia_smi
from pathlib import Path

def get_gpu_info():
    """Get GPU memory info for all available GPUs"""
    nvidia_smi.nvmlInit()
    gpu_count = nvidia_smi.nvmlDeviceGetCount()
    gpus = []
    
    for i in range(gpu_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpus.append({
            'id': i,
            'total_memory': info.total / 1024**3,  # Convert to GB
            'free_memory': info.free / 1024**3
        })
    
    return gpus

def get_optimal_config(gpus):
    """Calculate optimal configuration based on GPU count and memory"""
    gpu_count = len(gpus)
    total_gpu_mem = sum(gpu['total_memory'] for gpu in gpus)
    
    config = {
        'tensor_parallel_size': min(gpu_count, 4),  # Use up to 4 GPUs for tensor parallelism
        'gpu_memory_utilization': 0.85,  # Conservative memory usage
        'max_model_len': 8192,
        'block_size': 32,  # Optimal for most cases
        'swap_space': 8,  # 8GB swap space per GPU
        'num_scheduler_steps': 2,  # Increase scheduler steps
        'enable_chunked_prefill': True
    }
    
    if total_gpu_mem > 64:  # High memory setup
        config.update({
            'max_num_batched_tokens': 8192,
            'max_num_seqs': 256,
            'enable_prefix_caching': True
        })
    
    return config

def create_env_file():
    """Create environment file for API key"""
    env_path = Path.home() / '.vllm' / '.env'
    env_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not env_path.exists():
        api_key = input("Enter VLLM API key: ")
        with open(env_path, 'w') as f:
            f.write(f"VLLM_API_KEY={api_key}\n")
    
    return env_path

class DockerComposeGenerator:
    def __init__(self, config: dict, model_name: str):
        self.config = config
        self.model_name = model_name
        self.compose_path = Path('docker-compose.yml')

    def generate_command(self) -> list:
        cmd = [
            f"--model {self.model_name}",
            f"--tensor-parallel-size {self.config['tensor_parallel_size']}",
            f"--gpu-memory-utilization {self.config['gpu_memory_utilization']}",
            f"--max-model-len {self.config['max_model_len']}",
            f"--block-size {self.config['block_size']}",
            f"--swap-space {self.config['swap_space']}",
            f"--num-scheduler-steps {self.config['num_scheduler_steps']}",
            "--enable-chunked-prefill",
            f"--max-num-batched-tokens {self.config.get('max_num_batched_tokens', 4096)}",
            f"--max-num-seqs {self.config.get('max_num_seqs', 128)}"
        ]
        if self.config.get('enable_prefix_caching'):
            cmd.append("--enable-prefix-caching")
        return cmd

    def generate_compose(self) -> dict:
        return {
            'version': '3.8',
            'services': {
                'vllm': {
                    'image': 'vllm/vllm-openai:latest',
                    'runtime': 'nvidia',
                    'restart': 'unless-stopped',
                    'ports': ['8000:8000'],
                    'environment': ['NVIDIA_VISIBLE_DEVICES=all'],
                    'env_file': [f"{str(Path.home())}/.vllm/.env"],
                    'volumes': [
                        f"{str(Path.home())}/.cache/huggingface:/root/.cache/huggingface",
                        f"{str(Path.home())}/.vllm/.env:/root/.env"
                    ],
                    'ipc': 'host',
                    'command': ' '.join(self.generate_command()),
                    'deploy': {
                        'resources': {
                            'reservations': {
                                'devices': [{
                                    'driver': 'nvidia',
                                    'count': 'all',
                                    'capabilities': ['gpu']
                                }]
                            }
                        }
                    },
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    }
                }
            }
        }

    def save(self):
        compose_config = self.generate_compose()
        with open(self.compose_path, 'w') as f:
            yaml.safe_dump(compose_config, f, default_flow_style=False)
        return self.compose_path

def check_prerequisites():
    """Check if required software is installed"""
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
        subprocess.run(['docker','compose' , '--version'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("Docker or Docker Compose not installed")
    except FileNotFoundError:
        raise RuntimeError("Docker or Docker Compose not found in PATH")

def main():
    try:
        check_prerequisites()
        gpus = get_gpu_info()
        if not gpus:
            raise RuntimeError("No GPUs detected")
            
        config = get_optimal_config(gpus)
        model_name = "erax-ai/EraX-VL-2B-V1.5"
        
        # Generate docker-compose.yml
        generator = DockerComposeGenerator(config, model_name)
        compose_path = generator.save()
        
        # Create environment file
        env_path = create_env_file()
        
        # Start services
        subprocess.run(["docker", "compose", "up", "-d"], check=True)
        
        print(f"Docker Compose file generated at: {compose_path}")
        print(f"Configuration: {json.dumps(config, indent=2)}")
        print("Server running on http://localhost:8000")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()