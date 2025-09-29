import os
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import numpy as np # Added numpy
from rl_model_trainer import RLModelTrainer # Import RLModelTrainer
from dl_model_trainer import DLModelTrainer # Import DLModelTrainer for TARGET_COINS
from rl_environment import TradingEnv # Import TradingEnv
import scanner # Keep scanner for _calculate_breakout_levels_from_df
from market_regime_detector import MarketRegimeDetector

class MockSentimentAnalyzer:
    def analyze_market_sentiment(self, ticker: str):
        # For backtesting, always return a positive sentiment tuple
        return ('긍정적', '시뮬레이션 모드에서는 긍정적으로 가정')

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
                else:
                    print(f"Warning: Could not get current price for {ticker}. Excluding from total capital calculation.")
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
        pp, r1, s1, r2, s2, breakout_value = scanner._calculate_breakout_levels_from_df(df_daily)
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
        self.upbit = ccxt.upbit() # Public client for fetching OHLCV data
        self.optimization_results = []
        self.sentiment_analyzer = MockSentimentAnalyzer() # Instantiate mock sentiment analyzer
        
        self.dl_trainer = DLModelTrainer() # Instantiate DLModelTrainer
        self.dl_trainer.load_model() # Load the DL model for prediction

        self.rl_trainer = RLModelTrainer()
        self.rl_agent = self.rl_trainer.load_agent()
        if self.rl_agent is None:
            print("Warning: RL agent not loaded. Please train the agent first using '--mode train-rl'.")

    def load_historical_data(self):
        print("과거 데이터 로딩 중 (Feather 파일에서)... ")
        all_data = {}
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')

        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)

        for ticker in DLModelTrainer.TARGET_COINS: # Use TARGET_COINS from DLModelTrainer
            filename_feather = ticker.replace('/', '_') + '_1h.feather'
            filepath_feather = os.path.join(cache_dir, filename_feather)

            if not os.path.exists(filepath_feather):
                print(f"- {ticker} 캐시 파일 ({filepath_feather})을 찾을 수 없습니다. 건너뜁니다.")
                continue

            try:
                df = pd.read_feather(filepath_feather)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df[(df.index >= start_dt) & (df.index <= end_dt + timedelta(days=1, microseconds=-1))]

                if not df.empty:
                    all_data[ticker] = df
                    print(f"- {ticker} 데이터 로딩 완료 ({len(df)} 행)")
                else:
                    print(f"- {ticker} 데이터가 지정된 기간 내에 없습니다. 건너뜁니다.")
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

        print(f"--- Running RL-based Breakout Strategy Simulation ---")
            
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
                # Prepare data slice for the RL agent
                current_data_slice = {ticker: df[df.index <= timestamp] for ticker, df in all_ohlcv_data.items()}

                # --- 시장 체제 감지 (추가된 로직) ---
                regime_detector = MarketRegimeDetector()
                market_regime = 'Sideways' # 기본값
                if 'BTC/KRW' in all_ohlcv_data:
                    btc_df = all_ohlcv_data['BTC/KRW']
                    # 현재 타임스탬프 이전의 데이터만 사용하여 일봉 생성
                    btc_daily_df = btc_df[btc_df.index < timestamp]['close'].resample('D').last().to_frame()
                    market_regime = regime_detector.get_market_regime(btc_daily_df)
                print(f"[{timestamp}] 시장 체제 감지: {market_regime}")

                # --- DL 모델 기반 핫 코인 선정 로직 (수정됨) ---
                hot_coin_candidates = []
                # 시장 체제에 따라 매수 확률 임계값 동적 변경
                if market_regime == 'Bullish':
                    BUY_PROBABILITY_THRESHOLD = 0.55
                elif market_regime == 'Bearish':
                    BUY_PROBABILITY_THRESHOLD = 0.75
                else:  # Sideways
                    BUY_PROBABILITY_THRESHOLD = 0.65

                if self.dl_trainer.model is not None:
                    print(f"[{timestamp}] DL 모델로 핫 코인 스캔 중 (임계값: {BUY_PROBABILITY_THRESHOLD * 100}%)...")
                    for ticker in DLModelTrainer.TARGET_COINS:
                        df_1h = current_data_slice.get(ticker)
                        if df_1h is None or df_1h.empty:
                            continue

                        MIN_DATA_FOR_DL_PREDICTION = self.dl_trainer.sequence_length + 1
                        if len(df_1h) < MIN_DATA_FOR_DL_PREDICTION:
                            continue
                        
                        # [관망, 매수, 매도] 확률 예측
                        probabilities = self.dl_trainer.predict_proba(df_1h.copy())
                        
                        if probabilities is not None:
                            buy_probability = probabilities[1] # '매수' 확률
                            if buy_probability >= BUY_PROBABILITY_THRESHOLD:
                                hot_coin_candidates.append((ticker, buy_probability))
                                print(f"  -> 후보 발견: {ticker} (매수 확률: {buy_probability:.2f})")
                
                dl_selected_hot_coin_ticker = None
                best_buy_probability = -1
                if hot_coin_candidates:
                    # 매수 확률이 가장 높은 코인을 최종 선택
                    hot_coin_candidates.sort(key=lambda x: x[1], reverse=True)
                    dl_selected_hot_coin_ticker = hot_coin_candidates[0][0]
                    best_buy_probability = hot_coin_candidates[0][1]
                    print(f"  => 최종 후보 선정: {dl_selected_hot_coin_ticker} (매수 확률: {best_buy_probability:.2f})")
                
                # --- RL Agent Decision on DL-selected Hot Coin ---
                hot_coin_ticker = None
                if dl_selected_hot_coin_ticker:
                    print(f"DL Model selected {dl_selected_hot_coin_ticker} with buy probability: {best_buy_probability:.2f}. Now consulting RL agent.")
                    
                    if self.rl_agent is not None:
                        # Prepare observation for RL agent for the DL-selected coin
                        df_1h_for_rl = current_data_slice.get(dl_selected_hot_coin_ticker)
                        MIN_DATA_FOR_RL_OBSERVATION = TradingEnv(df=pd.DataFrame()).lookback_window
                        if len(df_1h_for_rl) < MIN_DATA_FOR_RL_OBSERVATION:
                            print(f"Not enough OHLCV data for {dl_selected_hot_coin_ticker} to evaluate with RL agent.")
                            continue # Skip this coin

                        df_1h_for_rl.fillna(0, inplace=True)
                        temp_env = TradingEnv(df=df_1h_for_rl.iloc[-MIN_DATA_FOR_RL_OBSERVATION:])
                        observation, _ = temp_env.reset()

                        # Predict action using the RL agent
                        action, _states = self.rl_agent.predict(observation, deterministic=True)
                        
                        if action == 1: # Action 1 corresponds to 'Buy'
                            hot_coin_ticker = dl_selected_hot_coin_ticker
                            print(f"RL agent confirms BUY for {hot_coin_ticker}.")
                        else:
                            print(f"RL agent recommends HOLD/SELL for {dl_selected_hot_coin_ticker} (Action: {action}). Skipping trade.")
                    else:
                        print(f"RL agent not loaded. Skipping RL decision for {dl_selected_hot_coin_ticker}.")
                else:
                    print("DL Model found no hot coins matching the criteria.")

                if hot_coin_ticker:
                    print(f"RL agent selected hot coin (Backtest): {hot_coin_ticker}. Now checking sentiment.")

                    # --- 시장 감성 분석 (모의 로직) ---
                    sentiment_text, sentiment_reason = self.sentiment_analyzer.analyze_market_sentiment(hot_coin_ticker)
                    print(f"Sentiment for {hot_coin_ticker}: {sentiment_text} (Reason: {sentiment_reason})")

                    if sentiment_text in ["매우 긍정적", "긍정적", "중립"]:
                        print(f"Sentiment is favorable ({sentiment_text}). Launching BreakoutTrader.")
                        current_price = current_data_slice[hot_coin_ticker].iloc[-1]['close']

                        # Calculate capital per trade dynamically
                        current_total_capital = self.portfolio.get_total_capital()
                        capital_per_trade = current_total_capital / self.max_concurrent_trades # Use max_concurrent_trades for allocation

                        # Launch BreakoutTrader
                        self.stats['breakout_trader_count'] += 1
                        df_daily = current_data_slice[hot_coin_ticker]['close'].resample('1D').ohlc().dropna()
                        self.active_traders[hot_coin_ticker] = SimulatedBreakoutTrader(hot_coin_ticker, current_price, self.portfolio, timestamp, df_daily, allocated_capital=capital_per_trade)
                    else:
                        print(f"Skipping {hot_coin_ticker} due to unfavorable sentiment: {sentiment}.")
                else:
                    print("RL agent found no hot coins to trade (Backtest).")

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
