import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging (1=INFO, 2=WARNING, 3=ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN custom operations
os.environ['OMP_NUM_THREADS'] = '1' # Limit OpenMP threads
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force TensorFlow to use CPU only

import argparse
import asyncio
import shutil
from dotenv import load_dotenv

# --- New High-Frequency System Modules --- #
from ccxt_downloader import CCXTDataDownloader, SCALPING_TARGET_COINS
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from live_trader import LiveTrader
from advanced_backtester import AdvancedBacktester
from rl_model_trainer import RLModelTrainer


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="High-Frequency Quant Scalping Bot")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["download", "preprocess", "train", "trade", "backtest", "train-rl", "validate", "simulate-commander"],
        help="""
        Operation mode:
        - 'download': Download 1-minute OHLCV data.
        - 'preprocess': Preprocess 1-minute data and create features.
        - 'train': Train the XGBoost model for micro-prediction.
        - 'trade': Start the high-frequency scalping live trader.
        - 'backtest': Run a simulation of the scalping strategy.
        - 'train-rl': Train the Reinforcement Learning agent.
        - 'validate': Validate a single model's performance.
        - 'simulate-commander': Run the full AI Commander strategy simulation.
        """,
    )
    # General arguments
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD) for download/backtest.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD) for download/backtest.",
    )
    parser.add_argument(
        "--tickers", nargs="+", help="List of target coins (e.g., BTC/KRW ETH/KRW)."
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=50000,
        help="Initial capital for trading or backtesting.",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to the model file for validation."
    )
    parser.add_argument("--output-path", type=str, help="Path to save the validation results JSON.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache directory before preprocessing data.")

    args = parser.parse_args()

    if args.end_date is None:
        from datetime import datetime

        args.end_date = datetime.now().strftime("%Y-%m-%d")

    # --- Mode Execution --- #
    if args.mode == "download":
        print("📥 Downloading 1-minute data...")
        downloader = CCXTDataDownloader()
        if args.tickers is None:
            args.tickers = SCALPING_TARGET_COINS # Use default if not provided
        for ticker in args.tickers:
            downloader.download_ohlcv(
                ticker, "1m", args.start_date, args.end_date
            )

    elif args.mode == "preprocess":
        print("⚙️ Preprocessing 1-minute data...")
        preprocessor = DataPreprocessor(target_coins=args.tickers)
        if args.clear_cache:
            cache_dir = preprocessor.cache_dir
            if os.path.exists(cache_dir):
                print(f"기존 캐시 디렉토리 {cache_dir}를 삭제합니다.")
                shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True) # Recreate the cache directory
        preprocessor.run()

    elif args.mode == "train":
        print("🤖 Training XGBoost model...")
        trainer = ModelTrainer(target_coins=args.tickers)
        trainer.train_model()

    elif args.mode == "trade":
        print("🚀 Starting high-frequency scalping trader...")
        trader = LiveTrader(capital=args.capital)
        await trader.run()

    elif args.mode == "backtest":
        print("🔍 Running backtest for high-frequency scalping strategy...")
        backtester = AdvancedBacktester(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital,
        )
        backtester.run_simulation()

    elif args.mode == "train-rl":
        print("🤖 Training Reinforcement Learning agent...")
        # Note: The ticker is hardcoded to BTC/KRW as an example.
        # This can be made configurable with another argparse argument if needed.
        rl_trainer = RLModelTrainer()
        rl_trainer.train_agent(total_timesteps=100000, ticker="BTC/KRW")
    
    elif args.mode == "validate":
        print(f"🔍 Validating model: {args.model_path}...")
        from portfolio_backtester import PortfolioBacktester
        validator = PortfolioBacktester(
            start_date=args.start_date,
            end_date=args.end_date,
            model_path=args.model_path,
            output_path=args.output_path,
            initial_capital=args.capital
        )
        validator.run_single_model_validation()

    elif args.mode == "simulate-commander":
        print("🚀 Running AI Commander simulation...")
        from portfolio_backtester import PortfolioBacktester
        commander_sim = PortfolioBacktester(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital
        )
        commander_sim.run_commander_simulation()


if __name__ == "__main__":
    asyncio.run(main())
