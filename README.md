
# AI Commander v2.0: Multi-Asset Deep Learning Trading Bot

This project is a sophisticated, multi-asset, deep learning-powered automated trading system for the Upbit cryptocurrency exchange.

It moves beyond simple rule-based strategies by leveraging a unified LSTM model to predict win probabilities across a dynamic universe of assets, coupled with a mathematical formula for optimal position sizing.

An integrated MLOps pipeline, featuring an "Opportunity Sentinel," ensures the system continuously learns from the market and improves over time.

---

## 🚀 Key Features

- **Dynamic Universe Selection**: Automatically trades the top 10 most liquid KRW pairs, re-evaluated periodically.
- **Deep Learning Brain**: An LSTM model predicts the probability of a price increase for each asset, serving as the core trading signal.
- **Mathematical Position Sizing**: Uses the Kelly Criterion formula to dynamically size positions based on the model's predicted win probability, optimizing for long-term capital growth.
- **Automated Risk Management**: All open positions are managed with a 10% Trailing Stop-Loss to maximize profit capture and limit drawdowns.
- **Self-Improving MLOps Loop**: A `sentinel.py` script runs on a schedule, detects missed trading opportunities (e.g., "V-Recovery" patterns), and automatically triggers a retraining pipeline to fine-tune the prediction model with the new data.
- **Fully Automated CI/CD**: Pushing to the `main` branch automatically triggers a pipeline that lints, backtests, and deploys the latest version of the bot to a live AWS EC2 server.

## 🏗️ System Architecture

The system is composed of four key modules:

1.  **`universe_manager.py`**: The Coin Selector. This script identifies the most liquid assets on Upbit to form the current trading universe.

2.  **`dl_predictor.py`**: The Deep Learning Brain. This module is responsible for building, training, and using the LSTM model to generate win-probability predictions.

3.  **`multi_asset_backtester.py`**: The Simulation Engine. This script simulates the entire multi-asset strategy, using the DL model for signals, Kelly Criterion for sizing, and a trailing stop-loss for exits. It provides a comprehensive report on performance.

4.  **`sentinel.py`**: The Auto-Improvement Engine. This MLOps tool runs periodically on the live server to find patterns the model may have missed, saves them as new training data, and triggers a model fine-tuning process.

## 🛠️ How to Use

### 1. Setup

- **Create Virtual Environment**: It is highly recommended to use a virtual environment. This project is tested with Python 3.11.
  ```bash
  python3.11 -m venv venv
  source venv/bin/activate
  ```

- **Install Dependencies**: Install all required packages.
  ```bash
  pip install -r requirements.txt
  # For Apple Silicon (M1/M2/M3), install tensorflow-metal for GPU acceleration
  pip install tensorflow-macos tensorflow-metal
  ```

### 2. Model Training & Backtesting

The `multi_asset_backtester.py` script is the main entry point for both training and testing.

- If no model (`data/v2_lstm_model.h5`) is found, the script will automatically enter **training mode** using historical data for the top coins.
- If a model already exists, it will skip training and proceed directly to **backtesting mode**.

```bash
# Ensure you are in the activated virtual environment
python multi_asset_backtester.py
```

### 3. Live Operation & Auto-Improvement

On a live server (e.g., AWS EC2), the `sentinel.py` script should be run periodically via a scheduler like `cron`.

- **Example Crontab Entry** (runs every hour at the 5-minute mark):
  ```crontab
  5 * * * * /path/to/your/project/run_sentinel.sh
  ```

This creates the automated feedback loop that allows the AI Commander to learn and evolve over time.
