PROFILES = {
    "local": {
        "model": "Qwen/Qwen3-4B-AWQ",
        "quantization": "awq",
        "dtype": "float16",
        "gpu_memory_utilization": 0.75,
        "max_model_len": 8192,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20
    },

    "server": {
        "model": "Qwen/Qwen3-8B-AWQ",
        "quantization": "awq",
        "dtype": "float16",
        "gpu_memory_utilization": 0.90,
        "max_model_len": 16384,
        "max_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20
    },
}