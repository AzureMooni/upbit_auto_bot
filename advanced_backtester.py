import pandas as pd
import ccxt
from datetime import datetime, timedelta
import scanner  # scanner.py import
import itertools
from model_trainer import ModelTrainer

# 가상 거래소 및 전략 클래스들
class SimulatedUpbitService:
    def __init__(self, initial_capital, all_ohlcv_data=None, current_timestamp=None):
        self.balance = {'KRW': initial_capital}
        self.fee = 0.0005  # 0.05%
        self.all_ohlcv_data = all_ohlcv_data
        self.current_timestamp = current_timestamp

    def get_balance(self):
        return self.balance

    def get_current_price(self, ticker: str):
        if self.all_ohlcv_data and self.current_timestamp and ticker in self.all_ohlcv_data:
            df = self.all_ohlcv_data[ticker]
            if self.current_timestamp in df.index:
                return df.loc[self.current_timestamp]['close']
        return None

    def get_total_capital(self):
        total_krw = self.balance['KRW']
        for currency, amount in self.balance.items():
            if currency != 'KRW' and amount > 0:
                ticker = f"{currency}/KRW"
                current_price = self.get_current_price(ticker)
                if current_price:
                    total_krw += amount * current_price
        return total_krw

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

    def set_current_timestamp(self, timestamp):
        self.current_timestamp = timestamp

class SimulatedBreakoutTrader:
    def __init__(self, ticker, entry_price, upbit_service, timestamp, df_daily, allocated_capital):
        self.ticker = ticker
        self.entry_price = entry_price
        self.upbit_service = upbit_service
        self.active = True
        self.qty = 0
        self.base_currency = ticker.split('/')[0]
        self.allocated_capital = allocated_capital

        # Calculate pivot points and R2 for this specific day
        pp, r1, s1, r2, s2, breakout_value = scanner._calculate_breakout_levels(df_daily)
        self.pp = pp
        self.r2 = r2

        # Initial Buy
        initial_investment = self.allocated_capital
        buy_order = self.upbit_service.create_market_buy_order(self.ticker, initial_investment, self.entry_price)
        if buy_order:
            self.qty = buy_order['amount']
            print(f"[{timestamp}] 🚀 변동성 돌파 진입: {self.ticker} at {self.entry_price:,.2f} KRW, 수량: {self.qty:.6f}. TP: {self.r2:,.2f}, SL: {self.pp:,.2f}")

    def update(self, new_price, timestamp):
        if not self.active:
            return None

        # Take Profit
        if new_price >= self.r2:
            sell_order = self.upbit_service.create_market_sell_order(self.ticker, self.qty, new_price)
            self.active = False
            profit = (new_price - self.entry_price) * self.qty
            print(f"[{timestamp}] 🎉 변동성 돌파 익절: {self.ticker} at {new_price:,.2f} KRW (R2). 수익: {profit:,.2f} KRW")
            return 'closed'

        # Stop Loss
        elif new_price <= self.pp:
            sell_order = self.upbit_service.create_market_sell_order(self.ticker, self.qty, new_price)
            self.active = False
            profit = (new_price - self.entry_price) * self.qty
            print(f"[{timestamp}] 🚨 변동성 돌파 손절: {self.ticker} at {new_price:,.2f} KRW (PP). 손실: {profit:,.2f} KRW")
            return 'closed'
        
        return 'active'

class AdvancedBacktester:
    def __init__(self, start_date, end_date, initial_capital, max_concurrent_trades: int = 1):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.max_concurrent_trades = max_concurrent_trades
        self.upbit = ccxt.upbit()
        self.optimization_results = []
        
        self.model_trainer = ModelTrainer()
        if not self.model_trainer.load_model():
            print("Warning: ML model or scaler files not found. Please train the model first using '--mode train'.")

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

        print(f"--- Running ML-based Breakout Strategy Simulation ---")
            
        # Reset portfolio and active traders for each new simulation run
        self.portfolio = SimulatedUpbitService(self.initial_capital, all_ohlcv_data=all_ohlcv_data)
        self.active_traders = {}
        self.stats = {
            'breakout_trader_count': 0
        }

        for timestamp in timeline:
            self.portfolio.set_current_timestamp(timestamp)

            # Only check for new trades once a day to avoid over-trading
            if timestamp.hour == 9 and not self.active_traders: # Simplified: only one trade at a time for simplicity
                # Prepare data slice for the scanner
                current_data_slice = {ticker: df[df.index <= timestamp] for ticker, df in all_ohlcv_data.items()}

                # 1. Find a hot coin using ML model
                hot_coin_ticker = None
                highest_buy_prob = -1

                if self.model_trainer.model and self.model_trainer.scaler:
                    for ticker, df_1h in current_data_slice.items():
                        if not ticker.endswith('/KRW'):
                            continue

                        if len(df_1h) < 150: # A safe margin for various indicators
                            continue
                        
                        try:
                            buy_prob = self.model_trainer.predict(df_1h.copy())
                            if buy_prob is not None and buy_prob > highest_buy_prob:
                                highest_buy_prob = buy_prob
                                hot_coin_ticker = ticker
                        except Exception as e:
                            print(f"Error predicting for {ticker} in backtesting: {e}")
                            continue

                if hot_coin_ticker and highest_buy_prob > 0.5: # Only consider if buy probability is reasonably high
                    print(f"ML model selected hot coin (Backtest): {hot_coin_ticker} with buy probability: {highest_buy_prob:.4f}")

                    current_price = current_data_slice[hot_coin_ticker].iloc[-1]['close']

                    # Calculate capital per trade dynamically
                    current_total_capital = self.portfolio.get_total_capital()
                    capital_per_trade = current_total_capital / self.max_concurrent_trades # Use max_concurrent_trades for allocation

                    # Launch BreakoutTrader
                    self.stats['breakout_trader_count'] += 1
                    df_daily = current_data_slice[hot_coin_ticker]['close'].resample('1D').ohlc().dropna()
                    self.active_traders[hot_coin_ticker] = SimulatedBreakoutTrader(hot_coin_ticker, current_price, self.portfolio, timestamp, df_daily, allocated_capital=capital_per_trade)
                else:
                    print("ML model found no hot coins with high buy probability (Backtest).")

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

        # After simulation, record the results
        final_asset_value = self.portfolio.get_balance()['KRW']
        total_return = (final_asset_value - self.initial_capital) / self.initial_capital * 100
        
        self.optimization_results.append({
            'final_asset_value': final_asset_value,
            'total_return': total_return,
            'breakout_trader_count': self.stats['breakout_trader_count']
        })
        self.print_final_report(final_asset_value, total_return)

    def print_final_report(self, final_asset_value, total_return):
        print(f"""
""" + "="*40)
        print("--- 시뮬레이션 결과 ---")
        print(f"기간: {self.start_date} ~ {self.end_date}")
        print(f"초기 자본: {self.initial_capital:,.0f} KRW")
        print(f"최종 자산: {final_asset_value:,.0f} KRW")
        print(f"총 수익률: {total_return:.2f} %")
        print(f"변동성 돌파 전략 실행 횟수: {self.stats['breakout_trader_count']}")
        print("="*40)

    def print_optimization_summary(self):
        # With ML-based single strategy, this becomes a single final report
        if not self.optimization_results:
            print("최적화 결과가 없습니다.")
            return

        best_result = self.optimization_results[0] # Only one result now

        print("""
""" + "█"*60)
        print("█████████████████ 최종 시뮬레이션 요약 █████████████████")
        print("█"*60)
        print(f"기간: {self.start_date} ~ {self.end_date}")
        print(f"초기 자본: {self.initial_capital:,.0f} KRW")

        print("\n--- 시뮬레이션 결과 ---")
        print(f"  최종 자산: {best_result['final_asset_value']:,.0f} KRW")
        print(f"  총 수익률: {best_result['total_return']:.2f} %")
        print(f"  변동성 돌파 전략 실행 횟수: {best_result['breakout_trader_count']}")
        print("█"*60)
        print("████████████████████████████████████████████████████")
