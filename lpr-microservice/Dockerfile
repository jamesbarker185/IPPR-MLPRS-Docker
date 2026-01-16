FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    tk \
    tcl \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Create non-root user and set home directory
RUN useradd -m -d /home/appuser appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /home/appuser/.paddleocr && \
    chown -R appuser:appuser /home/appuser

# Set environment variables for PaddleOCR
ENV HOME=/home/appuser
ENV PADDLE_HOME=/home/appuser/.paddleocr

USER appuser

EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
