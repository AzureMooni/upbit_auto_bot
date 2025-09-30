import pandas as pd
import numpy as np
import os
# TF_ENABLE_ONEDNN_OPTS=0 환경 변수 설정으로 mutex.cc 오류 방지
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from stable_baselines3 import PPO
from trading_env_simple import SimpleTradingEnv
from dl_model_trainer import DLModelTrainer
from sentiment_analyzer import SentimentAnalyzer

class PortfolioBacktester:
    def __init__(self, start_date: str, end_date: str, initial_capital: float = 10_000_000):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.target_coins = DLModelTrainer.TARGET_COINS
        self.cache_dir = 'cache'
        self.agents = self._load_specialist_agents()
        self.trade_log = []
        self.portfolio_history = []

    def _init_analyzer(self):
        print("\nGemini 정보 분석가를 준비합니다...")
        try:
            analyzer = SentimentAnalyzer()
            print("- 정보 분석가 준비 완료.")
            return analyzer
        except ValueError as e:
            print(f"- 경고: {e}")
            return None

    def _load_specialist_agents(self):
        agents = {}
        regimes = ['Bullish', 'Bearish', 'Sideways']
        print("\n훈련된 전문가 AI 에이전트들을 로드합니다...")
        
        try:
            dummy_df = pd.read_feather(os.path.join(self.cache_dir, f"{self.target_coins[0].replace('/', '_')}_1h.feather"))
            dummy_env = SimpleTradingEnv(dummy_df.select_dtypes(include=np.number))
        except Exception as e:
            print(f"오류: 에이전트 로드를 위한 더미 환경 생성 실패 - {e}")
            return None

        for regime in regimes:
            model_path = f"{regime.lower()}_market_agent.zip"
            if os.path.exists(model_path):
                print(f"  - [{regime}] 전문가 AI 로드 중...")
                agents[regime] = PPO.load(model_path, env=dummy_env)
            else:
                print(f"  - 경고: [{regime}] 전문가 모델({model_path})을 찾을 수 없습니다.")
        
        if not agents:
            print("오류: 어떤 전문가 AI 모델도 로드할 수 없습니다.")
            return None
        return agents

    def run_portfolio_simulation(self):
        if not self.agents:
            return

        print("\n백테스팅을 위해 캐시된 데이터를 로딩합니다...")
        all_data = {}
        for ticker in self.target_coins:
            cache_path = os.path.join(self.cache_dir, f"{ticker.replace('/', '_')}_1h.feather")
            if os.path.exists(cache_path):
                df = pd.read_feather(cache_path)
                df.set_index('timestamp', inplace=True)
                all_data[ticker] = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                print(f"  - {ticker} 데이터 로드 완료 ({len(all_data[ticker])}개)")

        if not all_data:
            print("오류: 백테스팅에 사용할 데이터가 없습니다.")
            return

        timeline = pd.date_range(self.start_date, self.end_date, freq='h')
        cash = self.initial_capital
        holdings = {ticker: 0.0 for ticker in self.target_coins}

        print(f"\n--- 🚀 포트폴리오 백테스팅 시작 ---")

        for now in timeline:
            # BTC 데이터가 없거나 현재 시간에 해당 데이터가 없으면 건너뜀
            if 'BTC/KRW' not in all_data or now not in all_data['BTC/KRW'].index:
                continue
            current_regime = all_data['BTC/KRW'].loc[now, 'regime']
            agent_to_use = self.agents.get(current_regime, self.agents.get('Sideways'))
            if agent_to_use is None: continue

            for ticker, df in all_data.items():
                if now not in df.index: continue

                # 현재 타임스탬프의 정수 인덱스 위치를 찾습니다.
                current_loc = df.index.get_loc(now)
                
                # 시작 인덱스가 0보다 작은 경우를 방지합니다.
                start_loc = max(0, current_loc - 50)
                
                # observation 데이터를 정수 인덱스 기반으로 슬라이싱하여 효율을 높입니다.
                observation_df = df.iloc[start_loc:current_loc]

                if len(observation_df) < 50: continue

                env_data = observation_df.select_dtypes(include=np.number)
                action, _ = agent_to_use.predict(env_data, deterministic=True)
                action = int(action) # NumPy 타입을 정수로 변환
                # print(f"  DEBUG: {now} | {ticker} | AI Predicted Action: {action}")

                current_price = df.loc[now, 'close']
                log_entry = {'timestamp': now, 'ticker': ticker, 'regime': current_regime, 'action': action, 'price': current_price}

                if action == 1: # Buy
                    buy_amount = cash * 0.05
                    if buy_amount > 5000:
                        holdings[ticker] += buy_amount / current_price
                        cash -= buy_amount
                        log_entry.update({'trade': 'BUY', 'amount_krw': buy_amount})
                        self.trade_log.append(log_entry)
                        print(f"  {now} | {ticker} | {current_regime} | BUY at {current_price:.2f}")
                elif action == 2: # Sell
                    if holdings[ticker] > 0:
                        sell_amount_coin = holdings[ticker]
                        cash += sell_amount_coin * current_price
                        holdings[ticker] = 0
                        log_entry.update({'trade': 'SELL', 'amount_coin': sell_amount_coin})
                        self.trade_log.append(log_entry)
                        print(f"  {now} | {ticker} | {current_regime} | SELL at {current_price:.2f}")
            
            # 현재 시점의 순자산 계산 (보유 코인 가치 + 현금)
            current_net_worth = cash
            for t, amount in holdings.items():
                if amount > 0 and t in all_data and now in all_data[t].index:
                    current_net_worth += amount * all_data[t].loc[now, 'close']
            self.portfolio_history.append({'timestamp': now, 'net_worth': current_net_worth})
        
        self._generate_final_report()

    def _generate_final_report(self):
        if not self.portfolio_history:
            print("성과를 분석할 데이터가 없습니다.")
            return

        print("\n--- 📊 최종 포트폴리오 성과 리포트 ---")
        history_df = pd.DataFrame(self.portfolio_history).set_index('timestamp')
        
        final_net_worth = history_df['net_worth'].iloc[-1]
        total_return = (final_net_worth - self.initial_capital) / self.initial_capital * 100
        print(f"- 총 수익률: {total_return:.2f}%")
        print(f"- 초기 자본: {self.initial_capital:,.0f} KRW")
        print(f"- 최종 자산: {final_net_worth:,.0f} KRW")

        history_df['peak'] = history_df['net_worth'].cummax()
        history_df['drawdown'] = (history_df['net_worth'] - history_df['peak']) / history_df['peak']
        max_drawdown = history_df['drawdown'].min() * 100
        print(f"- 최대 낙폭 (MDD): {max_drawdown:.2f}%")

        history_df['daily_return'] = history_df['net_worth'].pct_change()
        sharpe_ratio = (history_df['daily_return'].mean() / history_df['daily_return'].std()) * np.sqrt(365*24)
        print(f"- 샤프 지수 (시간봉 기준): {sharpe_ratio:.2f}")

        print("\n--- 👨‍🏫 전문가 AI별 거래 분석 ---")
        trade_df = pd.DataFrame(self.trade_log)
        if not trade_df.empty:
            # 'action' 컬럼은 정수이므로 groupby에 직접 사용 가능
            # 'trade' 컬럼이 이미 log_entry에 추가되므로 이를 사용
            print(trade_df.groupby(['regime', 'trade'])['ticker'].count().unstack(fill_value=0))
        else:
            print("거래 기록이 없습니다.")
        print("-------------------------------------")