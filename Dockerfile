FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    WORKSPACE_ROOT=/workspace \
    JUPYTER_PORT=8888

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    ca-certificates \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopengl0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /opt/neurcross

COPY pyproject.toml setup.py README.md ./
COPY neurcross ./neurcross
COPY models ./models
COPY quad_mesh ./quad_mesh
COPY utils ./utils

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        jupyterlab \
        notebook \
        ipykernel \
        ipywidgets \
        matplotlib \
        pandas \
        pymeshlab \
        kagglehub \
        huggingface_hub && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision torchaudio && \
    python -m pip install -e .

RUN mkdir -p /workspace/content

EXPOSE 8888

CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
