# --- STAGE 1: Builder ---
# This stage builds Python dependencies and runs training to generate models.
FROM python:3.12-slim AS builder

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install all Python dependencies
# We use requirements.txt here to ensure the final set of packages is correct.
COPY requirements.txt .
# The torch command is effectively handled by requirements.txt now, but we keep the extra index for it.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the source code
COPY . .

# Run build-time training to generate model artifacts
RUN mkdir -p /app/cache
RUN python foundational_model_trainer.py
RUN python specialist_trainer.py
RUN echo "Build-Time Training Complete. Model files generated."


# --- STAGE 2: Final ---
# This stage creates the final, lightweight image for runtime.
FROM python:3.12-slim AS final

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy the application code AND the generated models from the builder stage
COPY --from=builder /app /app

# Create a non-root user for security
RUN adduser --system --group appuser
# Ensure the app directory is owned by the new user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Set the entrypoint for the container
ENTRYPOINT ["python", "live_trader.py"]
