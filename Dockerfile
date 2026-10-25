# RFP Intelligence Platform: FastAPI (:8000) + Streamlit UI (:8501), started with the same
# command as locally (`python main.py up`). No secrets are copied into the image: the API key
# comes from .env at runtime (docker-compose.yml `env_file`), and .dockerignore excludes .env.
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Hugging Face models (embedder, reranker) are cached in the persistent data volume
    HF_HOME=/app/data/hf_cache

# Shared libraries needed by OpenCV / RapidOCR (Docling OCR) and PyTorch on slim Debian
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch (same version as the verified local setup) before the rest, so pip does not
# pull the multi-GB CUDA build; EMBEDDING_DEVICE=cpu is the configured device.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1 \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8501 8000

# `main.py up` stops both servers cleanly on Ctrl+C (KeyboardInterrupt), so stop with SIGINT.
STOPSIGNAL SIGINT
CMD ["python", "main.py", "up"]
