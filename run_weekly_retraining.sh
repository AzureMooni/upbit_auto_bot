#!/bin/bash
set -e

echo "[MLOps] Starting weekly full model retraining..."

# Activate virtual environment
cd /home/ec2-user/upbit_auto_bot || exit
source venv/bin/activate

# Run a temporary python script to trigger the training
python -c "
import pandas as pd
from dl_predictor import train_price_prediction_model

print('[MLOps] Loading full historical data from sample_data.pkl...')
data = pd.read_pickle('sample_data.pkl')
model_path = 'data/v2_lightgbm_model.joblib'

print('[MLOps] Initiating model training...')
train_price_prediction_model(data, model_path)

print('[SUCCESS] Weekly model retraining complete!')
"
