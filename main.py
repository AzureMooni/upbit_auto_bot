import argparse
import time
from core.exchange import UpbitService
from strategies.grid_trading import GridTrader
from scanner import find_hot_coin, get_dynamic_grid_prices
from core.backtester import Backtester # Backtester import 추가

def main():
    parser = argparse.ArgumentParser(description="Upbit Auto Trading Bot")
    parser.add_argument("--mode", type=str, required=True, choices=["grid", "backtest", "start-auto"],
                        help="Operation mode: 'grid' for live trading, 'backtest' for backtesting, 'start-auto' for fully automated trading.")
    parser.add_argument("--ticker", type=str, default="BTC/KRW", help="Trading ticker (e.g., BTC/KRW)")
    parser.add_argument("--lower-price", type=float, help="Lower price for grid trading")
    parser.add_argument("--upper-price", type=float, help="Upper price for grid trading")
    parser.add_argument("--grid-count", type=int, default=5, help="Number of grids (default: 5)")
    parser.add_argument("--order-amount-krw", type=float, default=10000, help="Order amount in KRW per grid (default: 10000)")
    parser.add_argument("--start-date", type=str, help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtesting (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.mode == "start-auto":
        print("🚀 자동매매 모드를 시작합니다. 거래할 코인을 탐색 중...")
        hot_coin_ticker = find_hot_coin()

        if hot_coin_ticker:
            print(f"✅ 자동매매 대상 코인 발견: {hot_coin_ticker}")
            lower_price, upper_price = get_dynamic_grid_prices(hot_coin_ticker)

            if lower_price is not None and upper_price is not None:
                print(f"📈 동적 가격 설정 완료: 상단 {upper_price:,.2f}원, 하단 {lower_price:,.2f}원")
                
                upbit_service = UpbitService()
                upbit_service.connect()
                
                grid_trader = GridTrader(
                    upbit_service,
                    hot_coin_ticker,
                    lower_price,
                    upper_price,
                    args.grid_count,
                    args.order_amount_krw
                )
                grid_trader.run(interval_seconds=5) # 5초 간격으로 실행
            else:
                print("❌ 동적 그리드 가격을 계산할 수 없습니다. 프로그램을 종료합니다.")
        else:
            print("🤖 거래에 적합한 코인을 찾지 못했습니다. 잠시 후 다시 시도합니다.")
            # 실제 운영에서는 여기에 일정 시간 대기 후 재시도 로직을 추가할 수 있습니다.
            # time.sleep(300) # 5분 대기 후 재시도 예시
            return # 코인을 찾지 못했으므로 종료

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
            args.order_amount_krw
        )
        grid_trader.run(interval_seconds=5)

    elif args.mode == "backtest":
        if not all([args.start_date, args.end_date, args.lower_price, args.upper_price]):
            print("Error: For 'backtest' mode, --start-date, --end-date, --lower-price, --upper-price are required.")
            return
        
        backtester = Backtester(args.ticker, args.start_date, args.end_date)
        backtester.run_test(args.lower_price, args.upper_price, args.grid_count, args.order_amount_krw)

if __name__ == "__main__":
    main()
