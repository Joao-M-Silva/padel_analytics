# Base: NVIDIA CUDA 12.4 + cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies (Python 3.10 is the Ubuntu 22.04 default — no PPA needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with the CUDA 12.4 wheel first (must come before requirements.txt)
RUN pip install --no-cache-dir \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install gdown for weight downloads at startup
RUN pip install --quiet gdown

# Create runtime directories that are gitignored
RUN mkdir -p cache weights/players_detection weights/ball_detection \
             weights/players_keypoints_detection weights/court_keypoints_detection

# Streamlit server settings
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Download weights on first start, then launch the app
CMD bash scripts/download_weights.sh && \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0
