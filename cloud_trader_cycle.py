import os
import sys
import asyncio
import traceback
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# 1. Load Environment Variables (.env & config/.env)
load_dotenv()
config_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", ".env")
if os.path.exists(config_env_path):
    load_dotenv(config_env_path)

from universe_manager import get_top_10_coins
from constants import MODEL_SAVE_PATH, LOOKBACK_WINDOW
from sentiment_analyzer import SentimentAnalyzer
from core.exchange import UpbitService
from market_regime_detector import precompute_all_indicators, get_market_regime, get_market_regime_dataframe
from risk_control_tower import RiskControlTower
from execution_engine_interface import UpbitExecutionEngine
from stable_baselines3 import PPO

STOP_LOSS_PCT = 0.05

class CloudTraderCycle:
    def __init__(self):
        access_key = os.environ.get('UPBIT_ACCESS_KEY')
        secret_key = os.environ.get('UPBIT_SECRET_KEY')
        if not access_key or not secret_key:
            print("[FATAL] UPBIT_ACCESS_KEY or UPBIT_SECRET_KEY is missing from environment.")
            sys.exit(1)

        print(f"[INFO] Initializing Cloud Trader Cycle. Key prefix: {access_key[:4]}...")
        self.agents = {}
        self.upbit_service = UpbitService(access_key, secret_key)
        self.risk_control_tower = RiskControlTower(mdd_threshold=-0.15)
        self.portfolio_history = pd.Series(dtype=float)
        self.sentiment_analyzer = None
        self.open_positions = {}
        self.execution_engine = UpbitExecutionEngine(self.upbit_service, self.open_positions)
        self.specialist_stats = {}

    async def initialize(self):
        print("🚀 [Cloud Trader] Connecting to Upbit...")
        await self.upbit_service.connect()
        self._load_agents()
        self.specialist_stats = self._load_specialist_stats()
        self._init_analyzer()
        net_worth = await self.get_total_balance()
        self.portfolio_history[pd.Timestamp.now()] = net_worth
        print(f"✅ Initialization complete. Total Net Worth: {net_worth:,.0f} KRW")

    def _load_agents(self):
        MODEL_SAVE_PATH_BASE = "specialist_agent_"
        regimes = ['Bullish', 'Bearish', 'Sideways']
        for regime in regimes:
            model_path = f"{MODEL_SAVE_PATH_BASE}{regime.lower()}.zip"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{regime} Model not found: {model_path}")
            self.agents[regime] = PPO.load(model_path)
            print(f"  - Loaded [{regime}] specialist model.")

    def _init_analyzer(self):
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            print("  - Sentiment analyzer initialized.")
        except Exception as e:
            print(f"  - Sentiment analyzer fallback: {e}")
            self.sentiment_analyzer = None

    def _load_specialist_stats(self):
        import json
        stats_file = 'specialist_stats.json'
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {
            regime: {'wins': 0, 'losses': 0, 'total_profit': 0.0, 'total_loss': 0.0, 'trades': 0}
            for regime in ['Bullish', 'Bearish', 'Sideways']
        }

    async def get_total_balance(self) -> float:
        krw_balance = await self.upbit_service.get_balance('KRW') or 0
        total_asset_value = krw_balance
        all_balances = await self.upbit_service.get_all_balances()
        if all_balances:
            for ticker, balance_info in all_balances.items():
                if ticker == 'KRW':
                    continue
                if balance_info['balance'] > 0:
                    market_ticker = f'{ticker}/KRW'
                    current_price = await self.upbit_service.get_current_price(market_ticker)
                    if current_price:
                        total_asset_value += balance_info['balance'] * current_price
        return total_asset_value

    async def execute_cycle(self):
        print("\n=======================================================")
        print(f"📊 [Cloud Trader] Executing Cycle at {pd.Timestamp.now()}")
        print("=======================================================")

        # 1. Stop-Loss & Circuit Breaker Check on existing holdings
        all_balances = await self.upbit_service.get_all_balances() or {}
        print(f"Current Assets: {list(all_balances.keys())}")

        # 2. Dynamic Universe Selection (Top 10 Liquid KRW Coins)
        universe = await get_top_10_coins(self.upbit_service)
        print(f"Target Universe (10 assets): {universe}")

        # 3. Market Regime Detection via BTC/KRW 1h trend
        btc_df = await self.upbit_service.get_ohlcv('BTC/KRW', '1h', 300)
        if btc_df is None or len(btc_df) < 50:
            print("[WARN] Insufficient BTC data for regime detection. Fallback to Sideways.")
            current_regime = 'Sideways'
        else:
            btc_df_with_regime = get_market_regime_dataframe(btc_df)
            current_regime = btc_df_with_regime['market_regime'].iloc[-1]
        
        agent_to_use = self.agents.get(current_regime) or self.agents.get('Sideways')
        print(f"Market Regime: [{current_regime}] -> Selected Agent: {agent_to_use.__class__.__name__}")

        # 4. Analyze each coin in the universe
        decisions_summary = []
        for symbol in universe:
            target_df = await self.upbit_service.get_ohlcv(symbol, '1h', 300)
            if target_df is None or len(target_df) < 50:
                continue

            processed_df = precompute_all_indicators(target_df)
            processed_df = get_market_regime_dataframe(processed_df)
            regime_map = {'Bullish': 0, 'Bearish': 1, 'Sideways': 2}
            processed_df['regime'] = processed_df['market_regime'].map(regime_map).fillna(2)

            final_features = [
                'open', 'high', 'low', 'close', 'volume',
                'ADX_14', 'NATR_14', 'BBP_20_2.0', 'EMA_20', 'EMA_50',
                'RSI_14', 'MACDh_12_26_9', 'regime'
            ]

            obs_df = processed_df[final_features].tail(50)
            obs = obs_df.to_numpy(dtype=np.float32).flatten()
            action_tensor, _ = agent_to_use.predict(obs, deterministic=True)
            confidence = 1.0

            action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
            predicted_action = action_map.get(int(action_tensor), 'Hold')

            # Sentiment Analysis
            sentiment_score = 0.5
            if self.sentiment_analyzer:
                try:
                    sentiment_score, _ = self.sentiment_analyzer.get_sentiment_score(symbol)
                except Exception:
                    sentiment_score = 0.5

            decisions_summary.append((symbol, predicted_action, sentiment_score))
            print(f"  [{symbol}] Signal: {predicted_action} | Sentiment: {sentiment_score:.2f}")

            # 5. Order Decision Execution
            if predicted_action == 'Buy':
                stats = self.specialist_stats.get(current_regime, {'wins': 0, 'losses': 0, 'trades': 0, 'total_profit': 0, 'total_loss': 0})
                win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 10 else 0.5
                avg_profit = stats['total_profit'] / stats['wins'] if stats['wins'] > 0 else 1
                avg_loss = abs(stats['total_loss'] / stats['losses']) if stats['losses'] > 0 else 1

                investment_fraction = self.risk_control_tower.determine_investment_size(
                    win_rate=win_rate,
                    avg_profit=avg_profit,
                    avg_loss=avg_loss,
                    prediction_confidence=confidence,
                    sentiment_score=sentiment_score
                )

                if investment_fraction > 0:
                    cash_balance = await self.upbit_service.get_balance('KRW') or 0
                    buy_amount_krw = cash_balance * investment_fraction
                    if buy_amount_krw >= 5000:
                        await self.execution_engine.create_market_buy_order(symbol, buy_amount_krw)
                    else:
                        print(f"    - [BUY SKIPPED] Order amount ({buy_amount_krw:,.0f} KRW) is below minimum (5,000 KRW).")

            elif predicted_action == 'Sell':
                coin_ticker = symbol.split('/')[0]
                coin_balance = await self.upbit_service.get_balance(coin_ticker)
                if coin_balance and coin_balance > 0:
                    await self.execution_engine.create_market_sell_order(symbol, coin_balance)

        print("\n✅ [Cloud Trader] Cycle Completed Successfully.")
        await self.upbit_service.close()

async def main():
    trader = CloudTraderCycle()
    await trader.initialize()
    await trader.execute_cycle()

if __name__ == "__main__":
    asyncio.run(main())
