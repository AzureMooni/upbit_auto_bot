# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for build and runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy all the application code
COPY . .

# --- Build-Time Training ---
# Run the complete data pipeline inside the build to generate all files
RUN mkdir -p /app/cache
RUN export UPBIT_ACCESS_KEY="DUMMY" && export UPBIT_SECRET_KEY="DUMMY" && python foundational_model_trainer.py

# Define the command to run your app
ENTRYPOINT ["python", "live_trader.py"]
