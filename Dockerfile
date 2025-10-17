# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for build and runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow for the Linux container environment
RUN pip install --no-cache-dir tensorflow

# Copy the rest of the application code
COPY . .

# Create a non-root user to run the application
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Re-copy files with correct ownership
COPY --chown=appuser:appuser . .

# Define the command to run your app
CMD ["python", "live_trader.py"]