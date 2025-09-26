import pandas as pd
import ccxt
from datetime import datetime, timedelta
import scanner  # scanner.py import
import itertools

# 가상 거래소 및 전략 클래스들
class SimulatedUpbitService:
    def __init__(self, initial_capital):
        self.balance = {'KRW': initial_capital}
        self.fee = 0.0005  # 0.05%

    def get_balance(self):
        return self.balance

    def create_market_buy_order(self, ticker, amount_krw, price):
        if self.balance['KRW'] >= amount_krw:
            coin_symbol = ticker.split('/')[0]
            qty = amount_krw / price
            self.balance['KRW'] -= amount_krw
            if coin_symbol not in self.balance:
                self.balance[coin_symbol] = 0
            self.balance[coin_symbol] += qty * (1 - self.fee)
            return {'symbol': ticker, 'price': price, 'amount': qty * (1 - self.fee)}
        return None

    def create_market_sell_order(self, ticker, qty, price):
        coin_symbol = ticker.split('/')[0]
        if coin_symbol in self.balance and self.balance[coin_symbol] >= qty:
            return_krw = qty * price * (1 - self.fee)
            self.balance[coin_symbol] -= qty
            self.balance['KRW'] += return_krw
            return {'symbol': ticker, 'price': price, 'amount': qty, 'cost': return_krw}
        return None

    def cancel_order(self, order_id):
        # In simulation, orders are executed instantly, so nothing to cancel
        pass

class SimulatedTrendFollower:
    def __init__(self, ticker, entry_price, upbit_service, timestamp, atr_multiplier):
        self.ticker = ticker
        self.entry_price = entry_price
        self.highest_price = entry_price
        self.upbit_service = upbit_service
        self.active = True
        self.qty = 0
        self.atr_multiplier = atr_multiplier
        self.trailing_stop_price = entry_price * (1 - (self.atr_multiplier * 0.01))  # Use atr_multiplier for dynamic stop-loss

        # Initial Buy
        initial_investment = self.upbit_service.get_balance()['KRW'] * 0.1 # Invest 10% of total capital
        buy_order = self.upbit_service.create_market_buy_order(self.ticker, initial_investment, self.entry_price)
        if buy_order:
            self.qty = buy_order['amount']
            print(f"[{timestamp}] ▲ 추세추종 진입: {self.ticker} at {self.entry_price:,.2f} KRW, 수량: {self.qty:.6f}")

    def update(self, new_price, timestamp):
        if not self.active:
            return None

        # Check for stop-loss
        if new_price < self.trailing_stop_price:
            sell_order = self.upbit_service.create_market_sell_order(self.ticker, self.qty, new_price)
            self.active = False
            profit = (new_price - self.entry_price) * self.qty
            print(f"[{timestamp}] ▼ 추세추종 종료: {self.ticker} at {new_price:,.2f} KRW, 수익: {profit:,.2f} KRW")
            return 'closed'

        # Update highest price and trailing stop
        if new_price > self.highest_price:
            self.highest_price = new_price
            new_stop_price = new_price * (1 - (self.atr_multiplier * 0.01))
            if new_stop_price > self.trailing_stop_price:
                self.trailing_stop_price = new_stop_price
                print(f"[{timestamp}] 📈 트레일링 스탑 조정: {self.trailing_stop_price:,.2f} KRW")
        
        return 'active'

class SimulatedRangeGridTrader:
    # This class would contain the grid trading simulation logic
    # For simplicity in this example, we will just log the start
    # A full implementation would be similar to the original backtester
    def __init__(self, ticker, upbit_service, timestamp):
        self.ticker = ticker
        self.upbit_service = upbit_service
        self.active = True
        print(f"[{timestamp}] ↕️ 박스권 그리드 시작: {self.ticker}")

    def update(self, new_price, timestamp):
        # In a real scenario, you'd check grid lines, execute trades, and check for stop-loss
        # For this example, we'll assume it runs for a set period and then closes.
        # This part needs to be fully implemented for accurate grid backtesting.
        pass

class AdvancedBacktester:
    def __init__(self, start_date, end_date, initial_capital,
                 ema_short_periods: list = None, ema_long_periods: list = None, atr_multipliers: list = None):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.upbit = ccxt.upbit()
        self.optimization_results = []

        self.ema_short_periods = ema_short_periods if ema_short_periods is not None else [20]
        self.ema_long_periods = ema_long_periods if ema_long_periods is not None else [60]
        self.atr_multipliers = atr_multipliers if atr_multipliers is not None else [2.0]

    def load_historical_data(self):
        print("과거 데이터 로딩 중...")
        all_data = {}
        markets = self.upbit.load_markets()
        krw_tickers = [m for m in markets if m.endswith('/KRW')]
        
        from_timestamp = self.upbit.parse8601(self.start_date + 'T00:00:00Z')
        limit = 1000 

        for ticker in krw_tickers[:30]: # Limit to top 30 for speed
            try:
                data = []
                since = from_timestamp
                while True:
                    ohlcv = self.upbit.fetch_ohlcv(ticker, '1h', since, limit)
                    if not ohlcv or ohlcv[-1][0] > self.upbit.parse8601(self.end_date + 'T23:59:59Z'):
                        break
                    data.extend(ohlcv)
                    since = ohlcv[-1][0] + 3600000 # advance 1 hour
                
                if data:
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    all_data[ticker] = df
                    print(f"- {ticker} 데이터 로딩 완료")
            except Exception as e:
                print(f"- {ticker} 데이터 로딩 실패: {e}")
        
        return all_data

    def run_simulation(self):
        all_ohlcv_data = self.load_historical_data()
        if not all_ohlcv_data:
            print("시뮬레이션할 데이터가 없습니다.")
            return

        # Create a unified timeline
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        timeline = pd.date_range(start=start_dt, end=end_dt, freq='h')

        # Iterate through all parameter combinations
        param_combinations = itertools.product(self.ema_short_periods, self.ema_long_periods, self.atr_multipliers)

        for ema_short, ema_long, atr_mult in param_combinations:
            print(f"--- Running simulation with EMA Short: {ema_short}, EMA Long: {ema_long}, ATR Multiplier: {atr_mult} ---")
            
            # Reset portfolio and active traders for each new simulation run
            self.portfolio = SimulatedUpbitService(self.initial_capital)
            self.active_traders = {}
            self.stats = {
                'trend_follower_count': 0,
                'range_grid_count': 0
            }

            for timestamp in timeline:
                # Only check for new trades once a day to avoid over-trading
                if timestamp.hour == 9 and not self.active_traders:
                    # Prepare data slice for the scanner
                    current_data_slice = {ticker: df[df.index <= timestamp] for ticker, df in all_ohlcv_data.items()}

                    # 1. Find a hot coin
                    hot_coin = scanner.find_hot_coin(current_data_slice, ema_short_period=ema_short, ema_long_period=ema_long)

                    if hot_coin:
                        # 2. Classify the market
                        market_status = scanner.classify_market(hot_coin, current_data_slice)
                        # print(f"[{timestamp}] 🕵️ 시장 진단: {hot_coin} -> {market_status}")

                        current_price = current_data_slice[hot_coin].iloc[-1]['close']

                        # 3. & 4. Run the appropriate strategy
                        if market_status == 'trending':
                            self.stats['trend_follower_count'] += 1
                            self.active_traders[hot_coin] = SimulatedTrendFollower(hot_coin, current_price, self.portfolio, timestamp, atr_multiplier=atr_mult)
                        elif market_status == 'ranging':
                            # The RangeGridTrader needs to be fully implemented for backtesting
                            # For now, it just logs the start
                            self.stats['range_grid_count'] += 1
                            self.active_traders[hot_coin] = SimulatedRangeGridTrader(hot_coin, self.portfolio, timestamp)

                # Update active traders with the current price
                traders_to_close = []
                for ticker, trader in self.active_traders.items():
                    if ticker in all_ohlcv_data and timestamp in all_ohlcv_data[ticker].index:
                        new_price = all_ohlcv_data[ticker].loc[timestamp]['close']
                        status = trader.update(new_price, timestamp)
                        if status == 'closed':
                            traders_to_close.append(ticker)
                
                # Remove closed traders
                for ticker in traders_to_close:
                    del self.active_traders[ticker]

            # After each simulation, record the results
            final_asset_value = self.portfolio.get_balance()['KRW']
            total_return = (final_asset_value - self.initial_capital) / self.initial_capital * 100
            
            self.optimization_results.append({
                'ema_short': ema_short,
                'ema_long': ema_long,
                'atr_multiplier': atr_mult,
                'final_asset_value': final_asset_value,
                'total_return': total_return,
                'trend_follower_count': self.stats['trend_follower_count'],
                'range_grid_count': self.stats['range_grid_count']
            })
            self.print_final_report(ema_short, ema_long, atr_mult, final_asset_value, total_return)

        self.print_optimization_summary()

    def print_final_report(self, ema_short, ema_long, atr_mult, final_asset_value, total_return):
        print(f"""
""" + "="*40)
        print("--- 시뮬레이션 결과 ---")
        print(f"기간: {self.start_date} ~ {self.end_date}")
        print(f"초기 자본: {self.initial_capital:,.0f} KRW")
        print(f"EMA Short: {ema_short}, EMA Long: {ema_long}, ATR Multiplier: {atr_mult}")
        print(f"최종 자산: {final_asset_value:,.0f} KRW")
        print(f"총 수익률: {total_return:.2f} %")
        print(f"추세추종 전략 실행 횟수: {self.stats['trend_follower_count']}")
        print(f"박스권 그리드 전략 실행 횟수: {self.stats['range_grid_count']}")
        print("="*40)

    def print_optimization_summary(self):
        if not self.optimization_results:
            print("최적화 결과가 없습니다.")
            return

        best_result = max(self.optimization_results, key=lambda x: x['total_return'])

        print("""
""" + "█"*60)
        print("█████████████████ 최적화 요약 결과 █████████████████")
        print("█"*60)
        print(f"기간: {self.start_date} ~ {self.end_date}")
        print(f"초기 자본: {self.initial_capital:,.0f} KRW")
        sorted_results = sorted(self.optimization_results, key=lambda x: x['total_return'], reverse=True)

        print("\n--- 모든 시뮬레이션 결과 (수익률 기준 내림차순) ---")
        print(f"{'EMA Short':<10} {'EMA Long':<10} {'ATR Mult':<10} {'최종 자산':<15} {'수익률(%)':<10}")
        print(f"{'--'*5:<10} {'--'*5:<10} {'--'*5:<10} {'--'*7:<15} {'--'*5:<10}")
        for res in sorted_results:
            print(f"{res['ema_short']:<10} {res['ema_long']:<10} {res['atr_multiplier']:<10.1f} {res['final_asset_value']:<15,.0f} {res['total_return']:<10.2f}")

        print("\n--- 최고의 매개변수 조합 ---")
        print(f"  EMA Short: {best_result['ema_short']}")
        print(f"  EMA Long: {best_result['ema_long']}")
        print(f"  ATR Multiplier: {best_result['atr_multiplier']}")
        print(f"  최종 자산: {best_result['final_asset_value']:,.0f} KRW")
        print(f"  최대 수익률: {best_result['total_return']:.2f} %")
        print("█"*60)
        print("████████████████████████████████████████████████████")