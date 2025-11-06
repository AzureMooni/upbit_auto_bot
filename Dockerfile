# --- STAGE 1: The 'Factory' (Build & Train) ---
FROM python:3.13-slim AS builder

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install HEAVY training libraries (CPU-Only)
WORKDIR /app
COPY requirements-builder.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
RUN pip install --no-cache-dir -r requirements-builder.txt

# 3. Copy all code and Run Training (Model Generation)
COPY . .
RUN mkdir -p /app/cache
RUN python foundational_model_trainer.py
RUN python specialist_trainer.py
RUN echo "Build-Time Training Complete. Model files generated."

# --- STAGE 2: The 'Store' (Final Lightweight Image - Under 500MB) ---
FROM python:3.12-slim AS final

WORKDIR /app

# Set pip target and python path
ENV PIP_TARGET=/app/pip_packages
ENV PYTHONPATH="/app/pip_packages:${PYTHONPATH:+:${PYTHONPATH}}"

# Copy and install requirements as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy pre-built pandas_ta from builder stage as root
COPY --from=builder /usr/local/lib/python3.13/site-packages/pandas_ta /usr/local/lib/python3.12/site-packages/pandas_ta

# Now, create the non-root user
RUN adduser --system --group appuser

# Copy the application files from the builder stage
COPY --from=builder /app/core /app/core
COPY --from=builder /app/strategies /app/strategies
COPY --from=builder /app/ccxt_downloader.py /app/ccxt_downloader.py
COPY --from=builder /app/execution_engine_interface.py /app/execution_engine_interface.py
COPY --from=builder /app/live_trader.py /app/live_trader.py
COPY --from=builder /app/market_regime_detector.py /app/market_regime_detector.py
COPY --from=builder /app/risk_control_tower.py /app/risk_control_tower.py
COPY --from=builder /app/risk_manager.py /app/risk_manager.py
COPY --from=builder /app/trading_env_simple.py /app/trading_env_simple.py
COPY --from=builder /app/universe_manager.py /app/universe_manager.py
COPY --from=builder /app/sentiment_analyzer.py /app/sentiment_analyzer.py
COPY --from=builder /app/foundational_model_trainer.py /app/foundational_model_trainer.py
COPY --from=builder /app/preprocessor.py /app/preprocessor.py
COPY --from=builder /app/constants.py /app/constants.py
COPY --from=builder /app/foundational_agent.zip /app/foundational_agent.zip
COPY --from=builder /app/specialist_agent_*.zip /app/
COPY --from=builder /app/specialist_stats.json /app/specialist_stats.json

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Final Entrypoint
ENTRYPOINT ["python", "live_trader.py"]