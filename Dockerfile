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

# Copy the rest of the application code
COPY . .

# [추가] AI 모델 및 필수 데이터 파일을 이미지에 복사합니다.
# 'models' 디렉토리와 'data' 디렉토리가 있다고 가정합니다.
# 만약 파일/폴더 이름이 다르다면 이 부분을 수정해야 합니다.
COPY models/ /app/models/
COPY data/ /app/data/
COPY specialist_stats.json /app/specialist_stats.json

# Create a non-root user to run the application
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Re-copy files with correct ownership
COPY --chown=appuser:appuser . .

# Define the command to run your app
ENTRYPOINT ["python", "live_trader.py"]