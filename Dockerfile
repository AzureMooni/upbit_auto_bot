# 1. Base Image
FROM python:3.11-slim

# 2. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Working Directory
WORKDIR /app

# 4. Install OPTIMIZED Python Dependencies (CPU-Only)
# --- FIX: Install CPU-only torch FIRST from the extra index ---
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# --- THEN install all other requirements from default PyPI ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pyjwt==2.3.0
RUN pip install pyupbit

# 5. Copy ALL Application Code
COPY . .

# 6. --- Build-Time Training (Stable Strategy) ---
RUN mkdir -p /app/cache
RUN export UPBIT_ACCESS_KEY="DUMMY" && export UPBIT_SECRET_KEY="DUMMY" && python foundational_model_trainer.py

# 7. Final Entrypoint
ENTRYPOINT ["python", "live_trader.py"]
