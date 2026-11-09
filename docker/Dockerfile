# Optimized Dockerfile for Document AI Invoice/Receipt KIE Service (CPU Production)
FROM python:3.10-slim

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Install lightweight CPU-only PyTorch (saves ~7GB of unnecessary NVIDIA CUDA bloat)
# 2. Install remaining application dependencies
# 3. Clean up build-essential and pip cache to keep image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /root/.cache/pip

# Create unprivileged application user with home directory & writable cache paths
RUN groupadd -g 1001 appuser && \
    useradd -m -u 1001 -g appuser appuser && \
    mkdir -p /app/models /app/logs /home/appuser/.paddleocr /home/appuser/.cache/huggingface /home/appuser/.cache/torch && \
    chown -R appuser:appuser /app /home/appuser && \
    chmod -R 777 /home/appuser

ENV HOME=/home/appuser \
    PADDLE_OCR_BASE_DIR=/home/appuser/.paddleocr \
    HF_HOME=/home/appuser/.cache/huggingface \
    TORCH_HOME=/home/appuser/.cache/torch

# Copy application code and configurations
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser pyproject.toml /app/

USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server via Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
