# Dockerfile for Python services (pipeline + api)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p /data/auto-datapulse /data/homelab-mlops /mlflow

# Default CMD (overridden in docker-compose for pipeline)
CMD ["python", "-m", "src.serving.app"]
