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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy ALL Application Code
COPY . .

# 6. --- Build-Time Training (REMOVED) ---

# 7. Final Entrypoint
ENTRYPOINT ["python", "live_trader.py"]
