import argparse
import asyncio
from dotenv import load_dotenv

# --- New Imports for RL Portfolio Bot ---
from rl_agent_trainer import RLAgentTrainer
from live_trader import LiveTrader

# --- Data Utility Imports ---
from ccxt_downloader import download_ohlcv_data
from preprocessor import DataPreprocessor
from dl_model_trainer import DLModelTrainer # Still needed for TARGET_COINS and as a prerequisite

async def main():
    """Main function to run the trading bot in different modes."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Upbit Portfolio RL Trading Bot")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["download", "preprocess", "train-dl", "train-rl", "start-live"],
        help="""
        Operation mode:
        - 'download': Download historical OHLCV data.
        - 'preprocess': Preprocess data and create cached features.
        - 'train-dl': (Prerequisite) Train the DL model for feature extraction/analysis.
        - 'train-rl': Train the main Portfolio RL Agent.
        - 'start-live': Start live trading with the trained Portfolio RL Agent.
        """
    )
    parser.add_argument("--start-date", type=str, help="Start date for data download/processing (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for data download/processing (YYYY-MM-DD)")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total timesteps for RL training.")

    args = parser.parse_args()

    if args.mode == "download":
        if not all([args.start_date, args.end_date]):
            parser.error("For 'download' mode, --start-date and --end-date are required.")
        print("📥 데이터 다운로드를 시작합니다...")
        download_ohlcv_data(args.start_date, args.end_date)

    elif args.mode == "preprocess":
        if not all([args.start_date, args.end_date]):
            parser.error("For 'preprocess' mode, --start-date and --end-date are required.")
        print("⚙️ 데이터 전처리 및 캐시 생성을 시작합니다...")
        preprocessor = DataPreprocessor(target_coins=DLModelTrainer.TARGET_COINS)
        preprocessor.run(args.start_date, args.end_date)

    elif args.mode == "train-dl":
        if not all([args.start_date, args.end_date]):
            parser.error("For 'train-dl' mode, --start-date and --end-date are required.")
        print("🧠 딥러닝(LSTM) 모델 훈련을 시작합니다...")
        trainer = DLModelTrainer()
        all_data = trainer.load_historical_data(args.start_date, args.end_date)
        if not all_data:
            print("학습할 데이터가 없습니다.")
            return
        trainer.train_model(all_data)

    elif args.mode == "train-rl":
        print("🤖 포트폴리오 강화학습 에이전트 훈련을 시작합니다...")
        trainer = RLAgentTrainer()
        trainer.train_agent(total_timesteps=args.timesteps)

    elif args.mode == "start-live":
        print("🚀 포트폴리오 RL 에이전트로 실시간 자동매매를 시작합니다...")
        trader = LiveTrader()
        await trader.run()

if __name__ == "__main__":
    asyncio.run(main())