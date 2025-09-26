import argparse
import time
from core.exchange import UpbitService
from strategies.grid_trading import GridTrader
from strategies.trend_follower import TrendFollower # 새로운 전략 임포트
from strategies.range_grid import RangeGridTrader # 새로운 전략 임포트
from scanner import find_hot_coin, get_dynamic_grid_prices, classify_market, find_hot_coin_live, get_dynamic_grid_prices_live, classify_market_live # classify_market 임포트
from core.backtester import Backtester
from advanced_backtester import AdvancedBacktester # AdvancedBacktester import 추가

# Helper function to parse comma-separated integers
def parse_int_list(arg):
    return [int(x) for x in arg.split(',')]

# Helper function to parse comma-separated floats
def parse_float_list(arg):
    return [float(x) for x in arg.split(',')]

def main():
    parser = argparse.ArgumentParser(description="Upbit Auto Trading Bot")
    parser.add_argument("--mode", type=str, required=True, choices=["grid", "backtest", "start-auto", "simulate"],
                        help="Operation mode: 'grid' for live trading, 'backtest' for backtesting, 'start-auto' for fully automated trading, 'simulate' for advanced backtesting.")
    parser.add_argument("--ticker", type=str, default="BTC/KRW", help="Trading ticker (e.g., BTC/KRW)")
    parser.add_argument("--lower-price", type=float, help="Lower price for grid trading")
    parser.add_argument("--upper-price", type=float, help="Upper price for grid trading")
    parser.add_argument("--grid-count", type=int, default=5, help="Number of grids (default: 5)")
    parser.add_argument("--order-amount-krw", type=float, default=10000, help="Order amount in KRW per grid (default: 10000)")
    parser.add_argument("--start-date", type=str, help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--ema-short-periods", type=parse_int_list, default="20",
                        help="Comma-separated list of short EMA periods for trend filter (default: 20)")
    parser.add_argument("--ema-long-periods", type=parse_int_list, default="60",
                        help="Comma-separated list of long EMA periods for trend filter (default: 60)")
    parser.add_argument("--atr-multipliers", type=parse_float_list, default="2.0",
                        help="Comma-separated list of ATR multipliers for dynamic trailing stop-loss (default: 2.0)")

    args = parser.parse_args()

    if args.mode == "start-auto":
        print("🚀 자동매매 모드를 시작합니다. 거래할 코인을 탐색 중...")
        upbit_service = UpbitService()
        upbit_service.connect() # 서비스 연결은 한 번만

        hot_coin_ticker = find_hot_coin_live(upbit_service.exchange, ema_short_period=args.ema_short_periods[0], ema_long_period=args.ema_long_periods[0]) # UpbitService의 exchange 객체 전달

        if hot_coin_ticker:
            print(f"✅ 자동매매 대상 코인 발견: {hot_coin_ticker}")
            
            market_type = classify_market_live(hot_coin_ticker, upbit_service.exchange) # 시장 분류
            print(f"📊 현재 시장 유형: {market_type}")

            if market_type == "trending":
                print(f"📈 추세장 감지! TrendFollower 전략을 시작합니다.")
                trend_follower = TrendFollower(
                    upbit_service,
                    hot_coin_ticker,
                    order_amount_krw=args.order_amount_krw, # TrendFollower에도 주문 금액 전달
                    atr_multiplier=args.atr_multipliers[0]
                )
                trend_follower.run(interval_seconds=5)
            elif market_type == "ranging":
                print(f"📉 횡보장 감지! RangeGridTrader 전략을 시작합니다.")
                lower_price, upper_price = get_dynamic_grid_prices_live(hot_coin_ticker, upbit_service.exchange) # UpbitService의 exchange 객체 전달

                if lower_price is not None and upper_price is not None:
                    print(f"📈 동적 가격 설정 완료: 상단 {upper_price:,.2f}원, 하단 {lower_price:,.2f}원")
                    range_grid_trader = RangeGridTrader(
                        upbit_service,
                        hot_coin_ticker,
                        lower_price,
                        upper_price,
                        args.grid_count,
                        args.order_amount_krw
                    )
                    range_grid_trader.run(interval_seconds=5)
                else:
                    print("❌ 동적 그리드 가격을 계산할 수 없습니다. 프로그램을 종료합니다.")
            else:
                print(f"⚠️ 현재 시장 유형({market_type})에 맞는 전략을 찾지 못했습니다. 잠시 후 다시 시도합니다.")
                # 실제 운영에서는 여기에 일정 시간 대기 후 재시도 로직을 추가할 수 있습니다.
                time.sleep(300) # 5분 대기 후 재시도 예시
                return # 적합한 전략을 찾지 못했으므로 종료
        else:
            print("🤖 거래에 적합한 코인을 찾지 못했습니다. 잠시 후 다시 시도합니다.")
            time.sleep(300)
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

    elif args.mode == "simulate":
        if not all([args.start_date, args.end_date]):
            print("Error: For 'simulate' mode, --start-date and --end-date are required.")
            return
        
        print(f"🚀 고급 시뮬레이션 백테스터를 시작합니다. 기간: {args.start_date} ~ {args.end_date}")
        # 테스트할 매개변수 리스트 정의
        ema_short_list = [10, 20, 30]
        ema_long_list = [50, 60, 100]
        atr_multiplier_list = [1.5, 2.0, 3.0]

        # args 객체에 리스트 할당 (AdvancedBacktester가 이를 사용하도록)
        args.ema_short_periods = ema_short_list
        args.ema_long_periods = ema_long_list
        args.atr_multipliers = atr_multiplier_list

        advanced_backtester = AdvancedBacktester(
            args.start_date,
            args.end_date,
            initial_capital=50000,
            ema_short_periods=args.ema_short_periods,
            ema_long_periods=args.ema_long_periods,
            atr_multipliers=args.atr_multipliers
        )
        advanced_backtester.run_simulation()

if __name__ == "__main__":
    main()
