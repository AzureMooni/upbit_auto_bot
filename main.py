import argparse
import time
from core.exchange import UpbitService
from strategies.grid_trading import GridTrader
from core.backtester import Backtester
from advanced_backtester import AdvancedBacktester # AdvancedBacktester import 추가
from portfolio_manager import PortfolioManager # PortfolioManager import 추가
from ccxt_downloader import download_ohlcv_data # ccxt_downloader import 추가

# Helper function to parse comma-separated integers
def parse_int_list(arg):
    return [int(x) for x in arg.split(',')]

# Helper function to parse comma-separated floats
def parse_float_list(arg):
    return [float(x) for x in arg.split(',')]

def main():
    parser = argparse.ArgumentParser(description="Upbit Auto Trading Bot")
    parser.add_argument("--mode", type=str, required=True, choices=["grid", "backtest", "start-auto", "simulate", "train", "download"],
                        help="Operation mode: 'grid' for live trading, 'backtest' for backtesting, 'start-auto' for fully automated trading, 'simulate' for advanced backtesting, 'train' for training the ML model, 'download' for downloading historical data.")
    parser.add_argument("--ticker", type=str, default="BTC/KRW", help="Trading ticker (e.g., BTC/KRW)")
    parser.add_argument("--lower-price", type=float, help="Lower price for grid trading")
    parser.add_argument("--upper-price", type=float, help="Upper price for grid trading")
    parser.add_argument("--grid-count", type=int, default=5, help="Number of grids (default: 5)")
    parser.add_argument("--order-amount-krw", type=float, default=10000, help="Order amount in KRW per grid (default: 10000)")
    parser.add_argument("--start-date", type=str, help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--max-concurrent-trades", type=int, default=3,
                        help="Maximum number of concurrent trades for simulation (default: 3)")
    parser.add_argument("--capital", type=float, default=50000,
                        help="Initial capital for simulation (default: 50000 KRW)")

    args = parser.parse_args()

    if args.mode == "start-auto":
        print("🚀 포트폴리오 자동매매 모드를 시작합니다.")
        # PortfolioManager 인스턴스화 및 실행
        manager = PortfolioManager(total_capital=1_000_000, max_concurrent_trades=3) # 예시 값
        manager.run(scan_interval_seconds=60) # 1분마다 스캔

    elif args.mode == "grid":
        if not all([args.lower_price, args.upper_price]):
            print("Error: For 'grid' mode, --lower-price and --upper-price are required.")
            return

        upbit_service = UpbitService()
        upbit_service.connect()
        grid_trader = GridTrader(
            upbit_service,
            args.ticker,
            args.lower_price,
            args.upper_price,
            args.grid_count,
            allocated_capital=args.order_amount_krw * args.grid_count # For compatibility, assume order_amount_krw is per grid
        )
        grid_trader.run(interval_seconds=5)

    elif args.mode == "backtest":
        if not all([args.start_date, args.end_date, args.lower_price, args.upper_price]):
            print("Error: For 'backtest' mode, --start-date, --end-date, --lower-price, --upper-price are required.")
            return
        
        backtester = Backtester(args.ticker, args.start_date, args.end_date)
        backtester.run_test(args.lower_price, args.upper_price, args.grid_count, args.order_amount_krw)

    elif args.mode == "simulate":
        if not all([args.start_date, args.end_date]):
            print("Error: For 'simulate' mode, --start-date and --end-date are required.")
            return
        
        print(f"🚀 고급 시뮬레이션 백테스터를 시작합니다. 기간: {args.start_date} ~ {args.end_date}")

        advanced_backtester = AdvancedBacktester(
            args.start_date,
            args.end_date,
            initial_capital=args.capital,
            max_concurrent_trades=args.max_concurrent_trades
        )
        advanced_backtester.run_simulation()

    elif args.mode == "train":
        if not all([args.start_date, args.end_date]):
            print("Error: For 'train' mode, --start-date and --end-date are required.")
            return
        
        print(f"🚀 머신러닝 모델 학습을 시작합니다. 기간: {args.start_date} ~ {args.end_date}")

        from model_trainer import ModelTrainer

        trainer = ModelTrainer()
        all_data = trainer.load_historical_data(args.start_date, args.end_date)
        
        if not all_data:
            print("학습할 데이터가 없습니다.")
            return

        trainer.train_model(all_data)

    elif args.mode == "download":
        if not all([args.start_date, args.end_date]):
            print("Error: For 'download' mode, --start-date and --end-date are required.")
            return
        
        target_coins = ["BTC/KRW", "ETH/KRW", "XRP/KRW"]
        download_ohlcv_data(args.start_date, args.end_date, target_coins)

if __name__ == "__main__":
    main()
