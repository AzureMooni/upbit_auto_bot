
import argparse
import asyncio
from dotenv import load_dotenv

# --- 모듈 임포트 --- #
from ccxt_downloader import download_ohlcv_data
from preprocessor import DataPreprocessor
from foundational_model_trainer import train_foundational_agent
from specialist_trainer import train_specialists
from portfolio_backtester import PortfolioBacktester
from ai_commander import AICommander
from commander_backtester import CommanderBacktester
from dl_model_trainer import DLModelTrainer

async def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI Quant Fund Command Center")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["download", "preprocess", "train-foundational", "train-specialists", "start-commander", "walk-forward", "simulate-commander", "train-dl", "train-rl"],
        help="""
        Operation mode:
        - 'start-commander': AI 총사령관을 실행하여 실시간 자동매매를 시작합니다.
        - 'simulate-commander': AI 총사령관의 동적 자산배분 로직을 시뮬레이션합니다.
        - 'walk-forward': 개별 투자 전략의 성능을 워크 포워드 방식으로 검증합니다.
        - 'download': OHLCV 데이터를 다운로드합니다.
        - 'preprocess': 데이터를 전처리하고 캐시를 생성합니다.
        - 'train-foundational': 마스터 AI를 훈련합니다.
        - 'train-specialists': 전문가 AI들을 훈련합니다 (전이학습).
        - 'train-dl': 딥러닝 모델을 훈련합니다.
        - 'train-rl': 강화학습 에이전트를 훈련합니다.
        """
    )
    # 인자 추가
    parser.add_argument("--start-date", type=str, default="2021-01-01", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--tickers", nargs='+', help="대상 코인 목록 (e.g., BTC/KRW ETH/KRW)")
    parser.add_argument("--timesteps", type=int, default=200000, help="RL 훈련 타임스텝")
    parser.add_argument("--capital", type=float, default=1_000_000, help="운용 총 자본금")
    parser.add_argument("--symbol", type=str, default="BTC/KRW", help="실시간 거래 대상 심볼")
    parser.add_argument("--train-months", type=int, default=24, help="워크 포워드 최적화의 훈련 기간 (월)")
    parser.add_argument("--validation-months", type=int, default=6, help="워크 포워드 최적화의 검증 기간 (월)")

    args = parser.parse_args()

    if args.end_date is None:
        from datetime import datetime
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    # --- 모드별 실행 --- #
    if args.mode == "download":
        print("📥 데이터 다운로드를 시작합니다...")
        download_ohlcv_data(args.start_date, args.end_date, tickers=args.tickers)

    elif args.mode == "preprocess":
        print("⚙️ 데이터 전처리 및 캐시 생성을 시작합니다...")
        preprocessor = DataPreprocessor(target_coins=args.tickers)
        preprocessor.run()

    elif args.mode == "train-foundational":
        print("🤖 마스터 AI 훈련을 시작합니다...")
        train_foundational_agent(total_timesteps=args.timesteps)

    elif args.mode == "train-specialists":
        print("🎓 전문가 AI 훈련(전이학습)을 시작합니다...")
        train_specialists(total_timesteps_per_specialist=args.timesteps // 4)

    elif args.mode == "start-commander":
        print("🚀 AI 총사령관 시스템을 가동합니다...")
        commander = AICommander(total_capital=args.capital)
        await commander.run()
        
    elif args.mode == "walk-forward":
        print("🤖 개별 전략 성능 검증을 위해 워크 포워드 최적화를 시작합니다...")
        backtester = PortfolioBacktester(start_date=args.start_date, end_date=args.end_date, initial_capital=args.capital)
        backtester.run_walk_forward_optimization(train_months=args.train_months, validation_months=args.validation_months)

    elif args.mode == "simulate-commander":
        print("🧭 AI 총사령관 의사결정 시뮬레이션을 시작합니다...")
        commander_backtester = CommanderBacktester(start_date=args.start_date, end_date=args.end_date, initial_capital=args.capital)
        commander_backtester.run_simulation()

    elif args.mode == "train-dl":
        print("🧠 딥러닝 모델 훈련을 시작합니다...")
        trainer = DLModelTrainer()
        historical_data = trainer.load_historical_data(args.start_date, args.end_date)
        if historical_data:
            trainer.train_model(historical_data)
        else:
            print("데이터가 없어 딥러닝 모델 훈련을 종료합니다.")

    elif args.mode == "train-rl":
        print("🤖 강화학습 에이전트 훈련을 시작합니다...")
        from rl_model_trainer import RLModelTrainer
        trainer = RLModelTrainer()
        trainer.train_agent(total_timesteps=args.timesteps)

if __name__ == "__main__":
    asyncio.run(main())
