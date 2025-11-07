# --- STAGE 1: Builder ---
# This stage builds Python dependencies and runs training to generate models.
FROM python:3.12-slim AS builder

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install all Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy
# The torch command is effectively handled by requirements.txt now, but we keep the extra index for it.
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the source code and data needed for training
COPY . .

# Run build-time training to generate model artifacts
# Note: These scripts use data from the 'data/' directory
RUN python foundational_model_trainer.py
RUN python specialist_trainer.py
RUN echo "Build-Time Training Complete. Model files generated."


# --- STAGE 2: Final (Optimized) ---
# This stage creates the final, lightweight image for runtime.
FROM python:3.12-slim AS final

WORKDIR /app

# Create a non-root user for security
RUN adduser --system --group appuser

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy only the necessary application code and artifacts for runtime
COPY --from=builder --chown=appuser:appuser /app/live_trader.py .
COPY --from=builder --chown=appuser:appuser /app/universe_manager.py .
COPY --from=builder --chown=appuser:appuser /app/constants.py .
COPY --from=builder --chown=appuser:appuser /app/trading_env_simple.py .
COPY --from=builder --chown=appuser:appuser /app/sentiment_analyzer.py .
COPY --from=builder --chown=appuser:appuser /app/market_regime_detector.py .
COPY --from=builder --chown=appuser:appuser /app/risk_control_tower.py .
COPY --from=builder --chown=appuser:appuser /app/execution_engine_interface.py .
COPY --from=builder --chown=appuser:appuser /app/preprocessor.py . # Needed by market_regime_detector
COPY --from=builder --chown=appuser:appuser /app/ccxt_downloader.py . # Needed by preprocessor
COPY --from=builder --chown=appuser:appuser /app/dl_model_trainer.py . # Needed by preprocessor

# Copy necessary directories
COPY --from=builder --chown=appuser:appuser /app/core ./core
COPY --from=builder --chown=appuser:appuser /app/strategies ./strategies

# Copy generated models and stats
COPY --from=builder --chown=appuser:appuser /app/specialist_agent_*.zip .
COPY --from=builder --chown=appuser:appuser /app/specialist_stats.json .

# Copy the sentinel model, preserving the directory structure
COPY --from=builder --chown=appuser:appuser /app/data/v2_lightgbm_model.joblib ./data/v2_lightgbm_model.joblib

# It's best practice to provide .env file during runtime, not bake it in
# COPY .env . 

# Switch to the non-root user
USER appuser

# Set the entrypoint for the container
ENTRYPOINT ["python", "live_trader.py"]