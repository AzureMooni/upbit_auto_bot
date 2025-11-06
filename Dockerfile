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
RUN export UPBIT_ACCESS_KEY="DUMMY" && export UPBIT_SECRET_KEY="DUMMY" && python foundational_model_trainer.py
RUN export UPBIT_ACCESS_KEY="DUMMY" && export UPBIT_SECRET_KEY="DUMMY" && python specialist_trainer.py
RUN echo "Build-Time Training Complete. Model files generated."

# --- STAGE 2: The 'Store' (Final Lightweight Image - Under 500MB) ---
FROM python:3.13-slim AS final

WORKDIR /app
RUN adduser --system --group appuser
USER appuser
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /var/lib/apt/lists/*
# 5. Copy ONLY the essential files and generated models from the 'Factory' stage
# The 'final' image will NOT contain the heavy torch/xgboost libraries.
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

# Copy the GENERATED files (the small "brain" and "memory")
COPY --from=builder /app/foundational_agent.zip /app/foundational_agent.zip
COPY --from=builder /app/specialist_agent_*.zip /app/
COPY --from=builder /app/specialist_stats.json /app/specialist_stats.json

# Final Entrypoint
ENTRYPOINT ["python", "live_trader.py"]