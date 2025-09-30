import asyncio
import pandas as pd
import os
from stable_baselines3 import PPO

from trading_env_simple import SimpleTradingEnv
from sentiment_analyzer import SentimentAnalyzer
from core.exchange import UpbitService
from dl_model_trainer import DLModelTrainer
import numpy as np

class LiveTrader:
    def __init__(self, symbol: str, capital: float):
        self.symbol = symbol # 현재는 단일 심볼만 지원, 향후 확장 가능
        self.initial_capital = capital
        self.agents = {}
        self.sentiment_analyzer = None
        self.upbit_service = None

    async def initialize(self):
        """비동기 초기화 메서드"""
        self.upbit_service = UpbitService()
        await self.upbit_service.connect()
        self._load_agents()
        self._init_analyzer()

    def _load_agents(self):
        print("\n훈련된 전문가 AI 에이전트들을 로드합니다...")
        regimes = ['Bullish', 'Bearish', 'Sideways']
        # 더미 환경 생성을 위해 임시 데이터프레임 사용
        dummy_df = pd.DataFrame(np.random.rand(100, 21), columns=[f'f{i}' for i in range(21)])
        dummy_env = SimpleTradingEnv(dummy_df)

        for regime in regimes:
            model_path = f"{regime.lower()}_market_agent.zip"
            if os.path.exists(model_path):
                print(f"  - [{regime}] 전문가 AI 로드 중...")
                self.agents[regime] = PPO.load(model_path, env=dummy_env)
        
        if not self.agents:
            raise Exception("오류: 어떤 전문가 AI 모델도 로드할 수 없습니다.")

    def _init_analyzer(self):
        print("\nGemini 정보 분석가를 준비합니다...")
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            print("- 정보 분석가 준비 완료.")
        except ValueError as e:
            print(f"- 경고: {e}")

    async def run(self):
        """비동기 실시간 거래 메인 루프"""
        print(f"\n-- 🚀 AI 에이전트 팀 실시간 거래 시작 ({self.symbol}) --")
        while True:
            try:
                # 1. 데이터 가져오기 및 시장 진단 (BTC 기준)
                btc_df = await self.upbit_service.get_ohlcv('BTC/KRW', '1h', 200)
                if btc_df is None: 
                    await asyncio.sleep(30)
                    continue
                
                # 현재 시장 상황 진단 (preprocessor.py 로직과 동일하게)
                short_sma = btc_df['close'].rolling(window=20).mean().iloc[-1]
                long_sma = btc_df['close'].rolling(window=50).mean().iloc[-1]
                current_regime = 'Sideways'
                if short_sma > long_sma * 1.01: current_regime = 'Bullish'
                elif short_sma < long_sma * 0.99: current_regime = 'Bearish'
                print(f"\n{pd.Timestamp.now()}: 현재 시장 진단: {current_regime}")

                # 2. 전문가 AI 선택
                agent_to_use = self.agents.get(current_regime, self.agents.get('Sideways'))
                if not agent_to_use:
                    print("경고: 사용할 AI 에이전트가 없습니다. 1분 후 재시도합니다.")
                    await asyncio.sleep(60)
                    continue
                print(f"담당 전문가: [{current_regime}] Agent")

                # 3. 거래 대상 코인 데이터 가져오기 및 AI 예측
                target_df = await self.upbit_service.get_ohlcv(self.symbol, '1h', 200)
                if target_df is None: 
                    await asyncio.sleep(30)
                    continue
                
                # 예측을 위한 데이터 준비 (preprocessor 로직과 유사하게)
                # 실제 운영 시에는 preprocessor와 완벽히 동일한 로직 필요
                from preprocessor import DataPreprocessor # 임시 사용
                temp_preprocessor = DataPreprocessor()
                processed_df = temp_preprocessor._generate_features(target_df)
                processed_df.dropna(inplace=True)
                env_data = processed_df.select_dtypes(include=np.number)

                if len(env_data) < 50:
                    print("관측 데이터가 부족하여 예측을 건너뜁니다.")
                    await asyncio.sleep(60)
                    continue

                action, _ = agent_to_use.predict(env_data.tail(50), deterministic=True)
                action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
                predicted_action = action_map.get(int(action), 'Hold')
                print(f"AI 예측 행동: {predicted_action}")

                # 4. 최종 의사결정
                if predicted_action == 'Buy' and self.sentiment_analyzer:
                    sentiment = self.sentiment_analyzer.analyze(self.symbol)
                    if sentiment == 'Positive':
                        print("✅ 최종 승인: 시장 감성이 긍정적이므로 매수 주문을 실행합니다.")
                        # await self.upbit_service.create_market_buy_order(self.symbol, ...)
                    else:
                        print("⚠️ 보류: 시장 감성이 긍정적이지 않으므로 매수를 보류합니다.")
                elif predicted_action == 'Sell':
                    print("✅ 매도 결정: 매도 주문을 실행합니다.")
                    # await self.upbit_service.create_market_sell_order(self.symbol, ...)
                
                # 10분 대기
                print("--- 10분 후 다음 사이클 시작 ---")
                await asyncio.sleep(600)

            except Exception as e:
                print(f"거래 루프 중 오류 발생: {e}")
                await asyncio.sleep(60)

async def main_live():
    trader = LiveTrader(symbol='BTC/KRW', capital=1000000)
    await trader.initialize()
    await trader.run()

if __name__ == '__main__':
    asyncio.run(main_live())