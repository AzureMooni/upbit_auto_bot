import sys
import asyncio
import pandas as pd
import os
import json
import torch
import numpy as np
from stable_baselines3 import PPO

# Load API keys from environment variables
access_key = os.getenv("UPBIT_ACCESS_KEY")
secret_key = os.getenv("UPBIT_SECRET_KEY")

if not access_key or not secret_key:
    print("[FATAL] API Keys (UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY) were not found in environment variables.")
    sys.exit(1)


# --- Local Module Imports ---
from trading_env_simple import SimpleTradingEnv
from sentiment_analyzer import SentimentAnalyzer
from core.exchange import UpbitService
from market_regime_detector import precompute_all_indicators
from risk_control_tower import RiskControlTower
from execution_engine_interface import UpbitExecutionEngine

class LiveTrader:
    """
    AI 퀀트 펀드의 지휘 본부.
    RiskControlTower를 통해 모든 거래를 통제하고, 시장 상황에 맞춰 전문가 AI를 선택하여 거래를 수행합니다.
    """
    def __init__(self, symbol: str, capital: float):
        self.symbol = symbol
        self.initial_capital = capital
        self.agents = {}

        # --- Core Components ---
        self.sentiment_analyzer = None
        self.upbit_service = None
        self.risk_control_tower = RiskControlTower(mdd_threshold=-0.15)
        self.execution_engine = None

        # --- Data & State ---
        self.specialist_stats = self._load_specialist_stats()
        self.portfolio_history = pd.Series(dtype=float)

    async def initialize(self):
        """비동기 초기화: 모든 하위 모듈을 준비합니다."""
        print("🚀 AI 퀀트 펀드 시스템 초기화를 시작합니다...")
        self.upbit_service = UpbitService()
        await self.upbit_service.connect()

        self.execution_engine = UpbitExecutionEngine(self.upbit_service)
        self._load_agents()
        self._init_analyzer()

        # 초기 포트폴리오 가치 설정
        initial_net_worth = await self.get_total_balance()
        self.portfolio_history[pd.Timestamp.now()] = initial_net_worth
        print("✅ 시스템 초기화 완료.")

    def _load_agents(self):
        print("\n- 훈련된 전문가 AI 에이전트들을 로드합니다...")
        regimes = ['Bullish', 'Bearish', 'Sideways']
        dummy_df = pd.DataFrame(np.random.rand(100, 21), columns=[f'f{i}' for i in range(21)])
        dummy_env = SimpleTradingEnv(dummy_df)

        for regime in regimes:
            model_path = f"{regime.lower()}_market_agent.zip"
            if os.path.exists(model_path):
                print(f"  - [{regime}] 전문가 AI 로드 완료.")
                self.agents[regime] = PPO.load(model_path, env=dummy_env)

        if not self.agents:
            raise Exception("오류: 어떤 전문가 AI 모델도 로드할 수 없습니다.")

    def _init_analyzer(self):
        print("\n- Gemini 정보 분석가를 준비합니다...")
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            print("  - 정보 분석가 준비 완료.")
        except ValueError as e:
            print(f"  - 경고: {e}")

    def _load_specialist_stats(self):
        stats_file = 'specialist_stats.json'
        print(f"\n- 과거 전문가 AI 성과({stats_file})를 로드합니다...")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                print("  - 성과 데이터 로드 완료.")
                return stats
        else:
            print("  - 경고: 성과 데이터 파일이 없습니다. 기본값으로 시작합니다.")
            return {
                regime: {'wins': 0, 'losses': 0, 'total_profit': 0.0, 'total_loss': 0.0, 'trades': 0}
                for regime in ['Bullish', 'Bearish', 'Sideways']
            }

    async def get_total_balance(self) -> float:
        """현금과 보유 코인의 가치를 합산하여 총 자산을 KRW로 반환합니다."""
        krw_balance = await self.upbit_service.get_balance('KRW') or 0
        total_asset_value = krw_balance

        all_balances = await self.upbit_service.get_all_balances()
        for ticker, balance_info in all_balances.items():
            if balance_info['balance'] > 0:
                market_ticker = f"{ticker}/KRW"
                current_price = await self.upbit_service.get_current_price(market_ticker)
                if current_price:
                    total_asset_value += balance_info['balance'] * current_price
        return total_asset_value

    async def run(self):
        """비동기 실시간 거래 메인 루프"""
        print(f"\n-- 🚀 AI 퀀트 펀드 실시간 운영 시작 ({self.symbol}) --")
        while True:
            try:
                # 1. 포트폴리오 상태 업데이트 및 서킷 브레이커 확인
                net_worth = await self.get_total_balance()
                self.portfolio_history[pd.Timestamp.now()] = net_worth
                if self.risk_control_tower.check_mdd_circuit_breaker(self.portfolio_history):
                    all_balances = await self.upbit_service.get_all_balances()
                    holdings_to_liquidate = {f"{ticker}/KRW": info['balance'] for ticker, info in all_balances.items() if info['balance'] > 0 and ticker != 'KRW'}
                    await self.execution_engine.liquidate_all_positions(holdings_to_liquidate)
                    print("🚨 모든 거래가 중단되었습니다. 시스템을 종료합니다.")
                    break

                # 2. 시장 분석 및 전문가 AI 선택
                btc_df = await self.upbit_service.get_ohlcv('BTC/KRW', '1h', 200)
                if btc_df is None:
                    await asyncio.sleep(30)
                    continue

                short_sma = btc_df['close'].rolling(window=20).mean().iloc[-1]
                long_sma = btc_df['close'].rolling(window=50).mean().iloc[-1]
                current_regime = 'Sideways'
                if short_sma > long_sma * 1.01:
                    current_regime = 'Bullish'
                elif short_sma < long_sma * 0.99:
                    current_regime = 'Bearish'
                print(f"\n{pd.Timestamp.now()}: 현재 시장 진단: {current_regime}")

                agent_to_use = self.agents.get(current_regime, self.agents.get('Sideways'))
                if not agent_to_use:
                    print("경고: 사용할 AI 에이전트가 없습니다.")
                    await asyncio.sleep(60)
                    continue
                print(f"  - 담당 전문가: [{current_regime}] Agent")

                # 3. 데이터 준비 및 AI 예측/확신도 계산
                target_df = await self.upbit_service.get_ohlcv(self.symbol, '1h', 200)
                if target_df is None:
                    await asyncio.sleep(30)
                    continue

                processed_df = precompute_all_indicators(target_df)
                if len(processed_df) < 50:
                    print("관측 데이터 부족")
                    await asyncio.sleep(60)
                    continue

                obs = processed_df.tail(50).to_numpy()
                obs_tensor = torch.as_tensor(obs).float()
                action_tensor, _ = agent_to_use.predict(obs, deterministic=True)

                # 행동에 대한 로그 확률(log_prob)을 통해 확신도 계산
                _, log_prob, _ = agent_to_use.policy.evaluate_actions(obs_tensor.unsqueeze(0), torch.as_tensor([action_tensor]))
                confidence = torch.exp(log_prob).item()

                action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
                predicted_action = action_map.get(int(action_tensor), 'Hold')
                print(f"  - AI 예측: {predicted_action} (확신도: {confidence:.2%})")

                # 4. 감성 분석
                sentiment_score, _ = self.sentiment_analyzer.get_sentiment_score(self.symbol)

                # 5. 위험 관리 위원회(RCT)에 최종 결정 요청
                if predicted_action == 'Buy':
                    stats = self.specialist_stats[current_regime]
                    win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 10 else 0.5 # 데이터 부족 시 50%로 간주
                    avg_profit = stats['total_profit'] / stats['wins'] if stats['wins'] > 0 else 1
                    avg_loss = stats['total_loss'] / stats['losses'] if stats['losses'] > 0 else 1

                    investment_fraction = self.risk_control_tower.determine_investment_size(
                        win_rate, avg_profit, avg_loss, confidence, sentiment_score
                    )

                    if investment_fraction > 0:
                        cash_balance = await self.upbit_service.get_balance('KRW') or 0
                        buy_amount_krw = cash_balance * investment_fraction
                        if buy_amount_krw > 5000:
                            await self.execution_engine.create_market_buy_order(self.symbol, buy_amount_krw)
                        else:
                            print("  - [EXEC] 주문 금액이 최소 기준(5,000 KRW) 미만입니다.")

                elif predicted_action == 'Sell':
                    coin_balance = await self.upbit_service.get_balance(self.symbol.split('/')[0])
                    if coin_balance and coin_balance > 0:
                        await self.execution_engine.create_market_sell_order(self.symbol, coin_balance)
                    else:
                        print("  - [EXEC] 매도할 코인이 없습니다.")

                print("--- 10분 후 다음 사이클 시작 ---")
                await asyncio.sleep(600)

            except Exception as e:
                print(f"거래 루프 중 치명적 오류 발생: {e}")
                await asyncio.sleep(60)

async def main_live():
    trader = LiveTrader(symbol='BTC/KRW', capital=1000000)
    await trader.initialize()
    await trader.run()

if __name__ == '__main__':
    asyncio.run(main_live())
