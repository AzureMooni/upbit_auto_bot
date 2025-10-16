#!/bin/bash
set -e

# --- Configuration ---
PROJECT_DIR="/home/ec2-user/upbit_auto_bot"
APP_SCRIPT="live_trader.py" # 최종적으로 실행할 스크립트
PYTHON_CMD="python3"

# --- 1. Install Dependencies ---
echo "[DEPLOY] Checking and installing system dependencies..."
# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt-get"
    sudo $PKG_MANAGER update -y
    sudo $PKG_MANAGER install -y git python3-pip python3-venv
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum"
    sudo $PKG_MANAGER update -y
    sudo $PKG_MANAGER install -y git python3 python3-pip
    # Amazon Linux 2 often requires python3-devel for venv
    sudo $PKG_MANAGER install -y python3-devel || true
else
    echo "[ERROR] Neither apt-get nor yum found. Cannot install dependencies." >&2
    exit 1
fi

# --- 2. Update Code ---
echo "[DEPLOY] Pulling latest code from GitHub..."
cd $PROJECT_DIR || exit
git pull

# --- 3. Setup Python Virtual Environment ---
echo "[DEPLOY] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# --- 4. Run Application in Background ---
echo "[DEPLOY] Running the application in the background..."

# Find and kill the old process if it's running
PROCESS_NAME=$APP_SCRIPT
if pgrep -f $PROCESS_NAME > /dev/null; then
    echo "[DEPLOY] An old process is running. Killing it now..."
    pkill -f $PROCESS_NAME
    sleep 2
fi

# Run the new script in the background using nohup
# All output will be redirected to logs/live_trader.log
nohup python $APP_SCRIPT > logs/live_trader.log 2>&1 &

echo "[SUCCESS] AI Commander is now running in the background."