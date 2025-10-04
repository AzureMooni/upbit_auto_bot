import argparse
import asyncio
from dotenv import load_dotenv

# --- New High-Frequency System Modules --- #
from ccxt_downloader import download_ohlcv_data
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from live_trader import LiveTrader
from advanced_backtester import AdvancedBacktester

async def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="High-Frequency Quant Scalping Bot")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["download", "preprocess", "train", "trade", "backtest"],
        help="""
        Operation mode:
        - 'download': Download 1-minute OHLCV data.
        - 'preprocess': Preprocess 1-minute data and create features.
        - 'train': Train the XGBoost model for micro-prediction.
        - 'trade': Start the high-frequency scalping live trader.
        - 'backtest': Run a simulation of the scalping strategy.
        """
    )
    # General arguments
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD) for download/backtest.")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD) for download/backtest.")
    parser.add_argument("--tickers", nargs='+', help="List of target coins (e.g., BTC/KRW ETH/KRW).")
    parser.add_argument("--capital", type=float, default=50000, help="Initial capital for trading or backtesting.")

    args = parser.parse_args()

    if args.end_date is None:
        from datetime import datetime
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    # --- Mode Execution --- #
    if args.mode == "download":
        print("📥 Downloading 1-minute data...")
        download_ohlcv_data(args.start_date, args.end_date, tickers=args.tickers, timeframe='1m')

    elif args.mode == "preprocess":
        print("⚙️ Preprocessing 1-minute data...")
        preprocessor = DataPreprocessor(target_coins=args.tickers)
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
        backtester = AdvancedBacktester(start_date=args.start_date, end_date=args.end_date, initial_capital=args.capital)
        backtester.run_simulation()

if __name__ == "__main__":
    asyncio.run(main())