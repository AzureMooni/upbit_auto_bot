# Dockerfile for upbit_auto_bot with Multi-stage Build

# --- Stage 1: Builder ---
# This stage installs all dependencies, including heavy ones for training.
FROM python:3.12-slim as builder

WORKDIR /app

# Install build tools and git
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install all dependencies for training and building
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Image ---
# This stage builds the lightweight final image for live trading.
FROM python:3.12-slim

WORKDIR /app

# Create a non-root user for security
RUN useradd --create-home appuser

# Copy only the runtime virtual environment from the builder stage
# We create a new venv and install only runtime requirements for a smaller image
COPY runtime-requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r runtime-requirements.txt

# Copy necessary source code for live trading
COPY main.py .
COPY live_trader.py .
COPY preprocessor.py . # Needed by live_trader
COPY core/ ./core/

# Copy trained models
COPY price_predictor.pkl .
COPY price_scaler.pkl .

# Set user and environment
USER appuser
ENV PATH="/opt/venv/bin:$PATH"

# Default command to show help
CMD ["python", "main.py", "--help"]