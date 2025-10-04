# Upbit High-Frequency Quant Scalping Bot

## 📖 Overview

This project is a high-frequency quantitative scalping auto-trading bot based on 1-minute candle data from the Upbit exchange. It uses an XGBoost machine learning model to predict ultra-short-term price movements and executes trades based on these predictions. It also includes a pipeline for training and deploying a Reinforcement Learning (RL) agent.

## ✨ Key Features

- **Data Collection:** Downloads 1-minute OHLCV data for specified periods and tickers from Upbit using `ccxt`.
- **Data Preprocessing:** Generates various technical analysis (TA) indicators such as RSI, MACD, etc., based on the downloaded data.
- **Model Training:** Trains an XGBoost classification model to predict price increases/decreases using the preprocessed data.
- **Backtesting:** Runs trading simulations on historical data using the trained model to evaluate performance.
- **Live Trading:** Automatically executes trades by receiving real-time data and making predictions with the model.
- **CI/CD:** The process from code quality checks, Docker image building, to AWS EC2 deployment is automated through GitHub Actions.

## 📂 Project Structure

```
/
├── .github/workflows/deploy.yml  # CI/CD Pipeline
├── .gitignore
├── Dockerfile                    # Configuration for building the Docker image
├── advanced_backtester.py        # Backtesting Engine
├── ccxt_downloader.py            # Data Downloader
├── data/                         # Raw OHLCV data (CSV)
├── cache/                        # Preprocessed data (Feather)
├── live_trader.py                # Live Trading Engine
├── main.py                       # Main executable file for the program
├── model_trainer.py              # XGBoost Model Trainer
├── preprocessor.py               # Data Preprocessor
├── price_predictor.pkl           # Trained XGBoost model
├── price_scaler.pkl              # Data scaler
└── requirements.txt              # List of Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/AzureMooni/upbit_auto_bot.git
    cd upbit_auto_bot
    ```

2.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3.  Install the `pandas-ta` library directly from the source (as per the workflow):
    ```bash
    pip install git+https://github.com/aarigs/pandas-ta.git
    ```

## 🛠️ Usage

You can run various modes using `main.py`.

- **Download Data:**
  ```bash
  python main.py --mode download --start-date 2025-09-01 --end-date 2025-10-01
  ```

- **Preprocess Data:**
  ```bash
  python main.py --mode preprocess
  ```

- **Train Model:**
  ```bash
  python main.py --mode train
  ```

- **Run Backtest:**
  ```bash
  python main.py --mode backtest --start-date 2025-09-01 --end-date 2025-10-01 --capital 50000
  ```

- **Start Live Trading:**
  ```bash
  python main.py --mode trade --capital 100000
  ```

## ⚙️ CI/CD Pipeline

This project uses GitHub Actions to build its CI/CD pipeline.

- **On Pull Request:** When a Pull Request is created for the `main` branch, a lint check using `ruff` is automatically executed.
- **On Push to Main:** When code is pushed to the `main` branch, after the lint check passes, a Docker image is built and pushed to Docker Hub, and the application is finally deployed to AWS EC2.
- **Scheduled Run:** Data collection, preprocessing, model training, and deployment are automatically run every Sunday at midnight UTC.