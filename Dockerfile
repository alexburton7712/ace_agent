# CUDA-enabled base image with Python
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
 
# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
 
# Install Python 3.12 and prerequisites (python3-pip removed here)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the primary system python interpreter
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Correct bootstrap URL for installing pip directly to Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
 
WORKDIR /app
 
# Install Python deps first (better layer caching)
COPY requirements.txt .

# Upgrade pip to absolute latest modern standards and build dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# FIXED: Replaced marketing link with the correct PyTorch CUDA package wheel repository
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
 
# Copy the app + prompt
COPY model.py .
COPY assistant_prompt.md .
 
# Hugging Face cache lives in a mounted volume (see docker-compose.yml)
ENV HF_HOME=/app/hf_cache
 
CMD ["python3", "model.py"]