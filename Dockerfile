# Use a base Python image
FROM python:3.12-slim-bookworm

# Set environment variable for non-interactive apt-get installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for PaddleOCR and OpenCV
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
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies first (for better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure
COPY . .

# Expose the port that Streamlit uses
EXPOSE 8501

# Set the default command to run Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
