FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgdal-dev \
    gdal-bin \
    python3-gdal \
    libhdf5-dev \
    libnetcdf-dev \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install flows2fim from GitHub releases
# Download and install flows2fim binary (v0.4.1 - Linux AMD64)
ARG FLOWS2FIM_VERSION=v0.4.1
RUN wget -q https://github.com/NGWPC/flows2fim-archive/releases/download/${FLOWS2FIM_VERSION}/flows2fim-linux-amd64.tar.gz \
    && tar -xzf flows2fim-linux-amd64.tar.gz \
    && mv flows2fim /usr/local/bin/flows2fim \
    && chmod +x /usr/local/bin/flows2fim \
    && rm flows2fim-linux-amd64.tar.gz

# Verify flows2fim installation
RUN flows2fim --version

# Copy application files
COPY *.py ./
COPY config.yaml ./

# Create output directories
RUN mkdir -p /data/input /data/output /data/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GDAL_DATA=/usr/share/gdal

# Default command
CMD ["python", "--version"]
