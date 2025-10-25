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

# [추가] Install specific library versions for compatibility
RUN pip install pyjwt==2.3.0
RUN pip install pyupbit

# Install TensorFlow for the Linux container environment
RUN pip install --no-cache-dir tensorflow

RUN mkdir -p /app/cache

# Copy the rest of the application code
COPY . .

# Run the training script to generate model files INSIDE the image
RUN export UPBIT_ACCESS_KEY="DUMMY" && export UPBIT_SECRET_KEY="DUMMY" && python foundational_model_trainer.py

# Define the command to run your app
ENTRYPOINT ["python", "live_trader.py"]