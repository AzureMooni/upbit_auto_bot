# 1. Base Image
FROM python:3.11-slim

# 2. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Working Directory
WORKDIR /app

# 4. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pyjwt==2.3.0
RUN pip install pyupbit
RUN pip install --no-cache-dir tensorflow

# 5. Copy ALL Application Code
COPY . .

# 6. --- Build-Time Training (The Fix) ---
# Create cache directory (fixes FileNotFoundError in preprocessor)
RUN mkdir -p /app/cache

# Run the trainer to fetch, preprocess, and train, generating all .pkl, .zip, and .json files
RUN export UPBIT_ACCESS_KEY="DUMMY" && export UPBIT_SECRET_KEY="DUMMY" && python foundational_model_trainer.py

# 7. Final Entrypoint
ENTRYPOINT ["python", "live_trader.py"]
