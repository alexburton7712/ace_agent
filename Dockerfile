# CUDA 13 development image
# vLLM 0.27.1 and your requirements.txt use CUDA 13.x packages.
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Prevent CUDA architecture auto-detection from causing unnecessary
# compilation problems. Change this if you know your GPU architecture.
# ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Install Python 3.12 and build tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    build-essential \
    ninja-build \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Upgrade packaging tools
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

# Install the EXACT pinned vLLM environment.
#
# Do NOT add the old CUDA 12.1 PyTorch index here.
# Your requirements already contain CUDA 13 packages.
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Hugging Face cache
ENV HF_HOME=/app/hf_cache

# Limit parallel compilation so Docker/container memory isn't exhausted
ENV MAX_JOBS=4

CMD ["python3", "main.py"]