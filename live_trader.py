import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from core.exchange import UpbitService
from rl_agent_trainer import RLAgentTrainer
from preprocessor import DataPreprocessor
from dl_model_trainer import DLModelTrainer # for TARGET_COINS
from rl_environment import PortfolioTradingEnv

class LiveTrader:
    """
    훈련된 강화학습 포트폴리오 에이전트를 사용하여 실시간 거래를 수행합니다.
    """
    def __init__(self, lookback_window: int = 50):
        self.upbit_service = UpbitService()
        self.agent_trainer = RLAgentTrainer()
        self.agent = None
        self.target_coins = DLModelTrainer.TARGET_COINS
        self.preprocessor = DataPreprocessor(target_coins=self.target_coins)
        self.lookback_window = lookback_window
        self.n_coins = len(self.target_coins)
        self.n_features = 0 # 데이터 로드 후 결정

    async def initialize(self) -> bool:
        """거래에 필요한 서비스 연결 및 모델 로드를 수행합니다."""
        await self.upbit_service.connect()
        
        # 에이전트 로드를 위해 더미 환경 생성
        dummy_data = await self._get_initial_data_for_env()
        if not dummy_data:
            print("에이전트 로드를 위한 초기 데이터 생성에 실패했습니다.")
            return False
        
        self.n_features = dummy_data[self.target_coins[0]].shape[1]
        dummy_env = PortfolioTradingEnv(dummy_data, lookback_window=self.lookback_window)
        self.agent = self.agent_trainer.load_agent(env=dummy_env)
        
        return self.agent is not None

    async def _get_initial_data_for_env(self) -> dict | None:
        """환경 초기화를 위해 모든 코인의 데이터를 가져옵니다."""
        print("환경 초기화를 위한 데이터 로딩 중...")
        all_data = {}
        for ticker in self.target_coins:
            df = await self.upbit_service.fetch_latest_ohlcv(ticker, '1h', self.lookback_window + 150)
            if df is None or len(df) < self.lookback_window + 100:
                print(f"{ticker}의 초기 데이터가 부족합니다.")
                return None
            processed_df = self.preprocessor._generate_features(df)
            processed_df.dropna(inplace=True)
            all_data[ticker] = processed_df
        return all_data

    async def _get_live_observation(self) -> dict | None:
        """실시간으로 관측(observation) 데이터를 수집하고 생성합니다."""
        # 1. 시장 데이터 수집 및 처리
        market_data_dict = await self._get_initial_data_for_env()
        if not market_data_dict:
            return None

        # 데이터 정렬 및 결합
        aligned_dfs = []
        for ticker in self.target_coins:
            df = market_data_dict[ticker].iloc[-self.lookback_window:].copy()
            df.columns = [f"{ticker}_{col}" for col in df.columns]
            aligned_dfs.append(df)
        combined_df = pd.concat(aligned_dfs, axis=1).ffill().bfill()

        market_obs = np.array([
            combined_df.iloc[:, i*self.n_features:(i+1)*self.n_features].values
            for i in range(self.n_coins)
        ])

        # 2. 포트폴리오 상태 수집
        balances = await self.upbit_service.get_balance()
        cash_balance = balances.get('KRW', 0)
        
        portfolio_values = []
        net_worth = cash_balance

        for ticker in self.target_coins:
            coin_symbol = ticker.split('/')[0]
            coin_balance = balances.get(coin_symbol, 0)
            current_price = await self.upbit_service.get_current_price(ticker)
            coin_value = coin_balance * (current_price or 0)
            portfolio_values.append(coin_value)
            net_worth += coin_value

        if net_worth == 0: net_worth = 1 # 0으로 나누기 방지

        portfolio_obs = np.concatenate(([cash_balance], portfolio_values)) / net_worth

        return {'portfolio': portfolio_obs.astype(np.float32), 'market': market_obs.astype(np.float32)}

    async def run(self, trade_interval_seconds: int = 3600):
        """실시간 거래 루프를 실행합니다."""
        print("🚀 포트폴리오 실시간 거래를 시작합니다.")
        if not await self.initialize():
            print("❌ Live Trader 초기화에 실패했습니다. 종료합니다.")
            return

        while True:
            try:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 다음 거래 사이클 대기 중...")
                await asyncio.sleep(trade_interval_seconds)
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 새로운 거래 사이클 시작.")
                
                # 1. 실시간 데이터로 관측(observation) 생성
                observation = await self._get_live_observation()
                if observation is None:
                    print("관측 데이터 생성 실패. 사이클을 건너뜁니다.")
                    continue

                # 2. AI 에이전트로부터 행동 결정
                action, _ = self.agent.predict(observation, deterministic=True)
                print(f"🤖 AI 에이전트 행동 제안: {action}")

                # 3. 행동 실행 (매수/매도)
                balances = await self.upbit_service.get_balance()
                cash_balance = balances.get('KRW', 0)

                for i, act in enumerate(action):
                    ticker = self.target_coins[i]
                    if act > 0.05: # 매수 (임계값 5% 이상)
                        buy_amount_krw = cash_balance * act
                        if buy_amount_krw > 5000: # 업비트 최소 주문 금액
                            print(f"  => [매수 실행] {ticker} | 규모: {buy_amount_krw:,.0f} KRW")
                            await self.upbit_service.create_market_buy_order(ticker, buy_amount_krw)
                    
                    elif act < -0.05: # 매도 (임계값 5% 이상)
                        coin_symbol = ticker.split('/')[0]
                        coin_balance = balances.get(coin_symbol, 0)
                        if coin_balance > 0:
                            sell_fraction = -act
                            sell_amount_coin = coin_balance * sell_fraction
                            current_price = await self.upbit_service.get_current_price(ticker)
                            if sell_amount_coin * (current_price or 0) > 5000:
                                print(f"  => [매도 실행] {ticker} | 수량: {sell_amount_coin:.6f} {coin_symbol}")
                                await self.upbit_service.create_market_sell_order(ticker, sell_amount_coin)

            except Exception as e:
                print(f"실시간 거래 루프 중 에러 발생: {e}")
                # 에러 발생 시 잠시 대기 후 재시도
                await asyncio.sleep(60)
