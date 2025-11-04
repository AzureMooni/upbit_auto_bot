#!/bin/bash
set -e

echo "[MLOps] Starting weekly full model retraining and deployment..."

# Navigate to the project directory
cd /home/ec2-user/upbit_auto_bot || { echo "ERROR: Project directory not found."; exit 1; }

# Pull the latest changes from the Git repository
echo "[MLOps] Pulling latest changes from Git..."
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Install any new or updated Python dependencies
echo "[MLOps] Installing/Updating Python dependencies..."
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -r build-requirements.txt # Specialist trainer needs this

# Run ccxt_downloader to get the latest historical data
echo "[MLOps] Downloading latest historical data..."
# Run in a subshell to set DOCKER_BUILD for ccxt_downloader.py
( DOCKER_BUILD=true python ccxt_downloader.py )

# Run the specialist trainer to retrain specialist models
echo "[MLOps] Retraining specialist models..."
python specialist_trainer.py

# Optionally, restart the Docker container with the new models
# This assumes the Docker image is built and push to ECR, and update.sh handles deployment.
# For simplicity in this script, we'll just re-run the update.sh.
echo "[MLOps] Redeploying the trading bot with new models..."
sudo bash update.sh

echo "[SUCCESS] Weekly model retraining and deployment complete!"