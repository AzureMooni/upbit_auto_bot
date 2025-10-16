#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# 1. Install Dependencies
# -----------------------
echo "[DEPLOY] Installing system dependencies (git, python3, venv)..."
sudo apt-get update -y
sudo apt-get install -y git python3-pip python3.11-venv

# 2. Update Code
# ----------------
echo "[DEPLOY] Pulling latest code from GitHub..."
cd /home/ec2-user/upbit_auto_bot || exit
git pull

# 3. Setup Python Virtual Environment
# -----------------------------------
echo "[DEPLOY] Setting up Python virtual environment..."
# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
# Activate and install/update requirements
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run Application
# ------------------
echo "[DEPLOY] Running the main script (sentinel.py)..."
# Now, run the script within the activated virtual environment
python sentinel.py

echo "[DEPLOY] Script execution finished."
