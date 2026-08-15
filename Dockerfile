# CUDA-enabled base image with Python
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the app + prompt
COPY model.py .
COPY assistant_prompt.md .

# Hugging Face cache lives in a mounted volume (see docker-compose.yml)
ENV HF_HOME=/app/hf_cache

CMD ["python3", "model.py"]