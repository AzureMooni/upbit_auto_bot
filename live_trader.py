import sys 
from dotenv import load_dotenv 
import os 
import asyncio 
import pandas as pd 
import numpy as np 
import torch 
import traceback 
import json 
from stable_baselines3 import PPO

# --- Core Module Imports ---
# This block fixes all previous ModuleNotFound/ImportErrors
try: 
    from universe_manager import get_top_10_coins 
    from foundational_model_trainer import train_foundational_agent 
    from preprocessor import DataPreprocessor 
    from trading_env_simple import SimpleTradingEnv 
    from sentiment_analyzer import SentimentAnalyzer 
    from core.exchange import UpbitService 
    from market_regime_detector import precompute_all_indicators, get_market_regime 
    from risk_control_tower import RiskControlTower 
    from execution_engine_interface import UpbitExecutionEngine 
except (ImportError, NameError) as e: 
    print(f'[FATAL] Failed to import core modules: {e}') 
    print(traceback.format_exc()) 
    sys.exit(1)

# --- 1. Load API Keys from .env file ---
load_dotenv() # .env 파일에서 환경 변수를 로드합니다.
access_key = os.getenv('UPBIT_ACCESS_KEY')
secret_key = os.getenv('UPBIT_SECRET_KEY')
if not access_key or not secret_key:
    print('[FATAL] UPBIT_ACCESS_KEY or UPBIT_SECRET_KEY not found in .env file.')
    sys.exit(1) 
print(f'[INFO] API Keys loaded successfully. Access Key starts with: {access_key[:4]}...')

# --- 2. Live Trader Class Definition ---
class LiveTrader: 
    """ AI 퀀트 펀드의 지휘 본부. RiskControlTower를 통해 모든 거래를 통제하고, 시장 상황에 맞춰 전문가 AI를 선택하여 거래를 수행합니다. """ 
    def __init__(self, capital: float): 
        self.initial_capital = capital 
        self.agents = {}

        # --- Core Components ---
        self.sentiment_analyzer = None
        self.upbit_service = UpbitService(access_key, secret_key)
        self.risk_control_tower = RiskControlTower(mdd_threshold=-0.15)
        self.execution_engine = UpbitExecutionEngine(self.upbit_service)
        
        # --- Data & State ---
        self.specialist_stats = self._load_specialist_stats()
        self.portfolio_history = pd.Series(dtype=float)

    async def initialize(self):
        """비동기 초기화: 모든 하위 모듈을 준비합니다."""
        print('🚀 AI 퀀트 펀드 시스템 초기화를 시작합니다...')
        await self.upbit_service.connect()
        self._load_agents()
        self._init_analyzer()
        
        # 초기 포트폴리오 가치 설정
        initial_net_worth = await self.get_total_balance()
        self.portfolio_history[pd.Timestamp.now()] = initial_net_worth
        print('✅ 시스템 초기화 완료.')

    def _load_agents(self):
        """
        'Train on First Boot' 로직:
        기초 모델을 로드합니다. 파일이 없으면, 훈련을 실행하여 생성합니다.
        """
        print('\n- 훈련된 전문가 AI 에이전트들을 로드합니다...')
        regimes = ['Bullish', 'Bearish', 'Sideways']
        model_path = 'foundational_agent.zip' # 단일 기초 모델
        dummy_env = None

        # 1. 'Train on First Boot' Logic
        if not os.path.exists(model_path):
            print(f'--- 경고: 훈련된 AI 모델({model_path})을 찾을 수 없습니다. ---')
            print('--- 최초 1회 훈련을 시작합니다... (최대 20분 소요) ---')

            # DUMMY 키를 설정하여 훈련 스크립트 실행 (환경 변수 사용)
            os.environ['UPBIT_ACCESS_KEY'] = 'DUMMY_KEY'
            os.environ['UPBIT_SECRET_KEY'] = 'DUMMY_KEY'
            
            try:
                # 훈련 실행
                train_foundational_agent(total_timesteps=150000) # 훈련 시간
            except Exception as e:
                print('[FATAL] 훈련 중 치명적인 오류 발생:')
                print(traceback.format_exc())
                raise e # 훈련에 실패하면 봇을 중지
            
            print('--- 훈련 완료! 에이전트를 다시 로드합니다. ---')

        # 2. Load the single foundational model
        if os.path.exists(model_path):
            try:
                # 로딩용 더미 환경 생성
                dummy_df = pd.DataFrame(np.random.rand(100, 21), columns=[f'f{i}' for i in range(21)])
                dummy_env = SimpleTradingEnv(dummy_df)
            except NameError:
                dummy_env = None # SimpleTradingEnv가 없으면 None으로 로드
            except Exception as e:
                print(f'[WARN] Dummy env for loading failed: {e}')
                dummy_env = None

            print(f'  - [Foundational] {model_path} 로드 시도...')
            foundational_model = PPO.load(model_path, env=dummy_env)
            
            # Assign the SAME foundational model to ALL regimes
            for regime in regimes:
                self.agents[regime] = foundational_model
            print(f'  - 모든 시장({regimes})에 기본 모델을 성공적으로 할당했습니다.')
        else:
            raise Exception(f'오류: 훈련을 시도했으나, AI 모델 파일({model_path})을 생성하지 못했습니다.')

    def _init_analyzer(self):
        print('\n- Gemini 정보 분석가를 준비합니다...')
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            print('  - 정보 분석가 준비 완료.')
        except ValueError as e:
            print(f'  - 경고: {e} (Gemini API 키가 없어 감성 분석을 건너뜁니다.)')
        except NameError as e:
            print(f'  - 경고: {e} (SentimentAnalyzer 모듈이 없어 감성 분석을 건너뜁니다.)')

    def _load_specialist_stats(self):
        stats_file = 'specialist_stats.json'
        print(f'\n- 과거 전문가 AI 성과({stats_file})를 로드합니다...')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                print('  - 성과 데이터 로드 완료.')
                return stats
        else:
            print('  - 경고: 성과 데이터 파일이 없습니다. 기본값으로 시작합니다.')
            return {
                regime: {'wins': 0, 'losses': 0, 'total_profit': 0.0, 'total_loss': 0.0, 'trades': 0}
                for regime in ['Bullish', 'Bearish', 'Sideways']
            }

    async def get_total_balance(self) -> float:
        """현금과 보유 코인의 가치를 합산하여 총 자산을 KRW로 반환합니다."""
        krw_balance = await self.upbit_service.get_balance('KRW') or 0
        total_asset_value = krw_balance
        
        all_balances = await self.upbit_service.get_all_balances()
        if not all_balances:
            print('[WARN] get_total_balance: 전체 잔고 정보를 가져오는 데 실패했습니다.')
            return krw_balance

        for ticker, balance_info in all_balances.items():
            if balance_info['balance'] > 0:
                market_ticker = f'KRW-{ticker}' # Ticker 형식 수정 (e.g., KRW-BTC)
                current_price = await self.upbit_service.get_current_price(market_ticker)
                if current_price:
                    total_asset_value += balance_info['balance'] * current_price
        return total_asset_value

    async def run(self):
        """비동기 실시간 거래 메인 루프"""
        print('\n-- 🚀 AI 퀀트 펀드 실시간 운영 시작 --')
        while True:
            try:
                # 1. 포트폴리오 상태 업데이트 및 서킷 브레이커 확인
                net_worth = await self.get_total_balance()
                self.portfolio_history[pd.Timestamp.now()] = net_worth
                if self.risk_control_tower.check_mdd_circuit_breaker(self.portfolio_history):
                    all_balances = await self.upbit_service.get_all_balances()
                    holdings_to_liquidate = {f'KRW-{ticker}': info['balance'] for ticker, info in all_balances.items() if info['balance'] > 0 and ticker != 'KRW'}
                    await self.execution_engine.liquidate_all_positions(holdings_to_liquidate)
                    print('🚨 모든 거래가 중단되었습니다. 시스템을 종료합니다.')
                    break

                # 2. 거래 유니버스 결정
                universe = get_top_10_coins()
                
                # 3. 각 자산에 대한 거래 결정
                for symbol in universe:
                    print(f'\n{pd.Timestamp.now()}: [{symbol}] 분석 시작...')
                    
                    # 3a. 시장 분석 및 전문가 AI 선택
                    btc_df = await self.upbit_service.get_ohlcv('KRW-BTC', '1h', 200)
                    if btc_df is None: continue
                    
                    short_sma = btc_df['close'].rolling(window=20).mean().iloc[-1]
                    long_sma = btc_df['close'].rolling(window=50).mean().iloc[-1]
                    current_regime = 'Sideways'
                    if short_sma > long_sma * 1.01: current_regime = 'Bullish'
                    elif short_sma < long_sma * 0.99: current_regime = 'Bearish'
                    
                    agent_to_use = self.agents.get(current_regime, self.agents.get('Sideways'))
                    if not agent_to_use:
                        print(f'경고: [{symbol}]을(를) 담당할 AI 에이전트가 없습니다.')
                        continue
                    print(f'  - 시장 진단: {current_regime}, 담당 전문가: [Foundational] Agent')

                    # 3b. 데이터 준비 및 AI 예측
                    target_df = await self.upbit_service.get_ohlcv(symbol, '1h', 200)
                    if target_df is None: continue
                    
                    processed_df = precompute_all_indicators(target_df) # preprocessor.py에서 임포트한 함수가 아님
                    if len(processed_df) < 50:
                        print('  - 관측 데이터 부족')
                        continue

                    obs = processed_df.tail(50).to_numpy()
                    obs_tensor = torch.as_tensor(obs).float()
                    action_tensor, _ = agent_to_use.predict(obs, deterministic=True)
                    
                    _, log_prob, _ = agent_to_use.policy.evaluate_actions(obs_tensor.unsqueeze(0), torch.as_tensor([action_tensor]))
                    confidence = torch.exp(log_prob).item()
                    
                    action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
                    predicted_action = action_map.get(int(action_tensor), 'Hold')
                    print(f'  - AI 예측: {predicted_action} (확신도: {confidence:.2%})')

                    # 3c. 감성 분석
                    sentiment_score = 0.5 # 기본값
                    if self.sentiment_analyzer:
                        sentiment_score, _ = self.sentiment_analyzer.get_sentiment_score(symbol)
                    
                    # 3d. 위험 관리 위원회(RCT)에 최종 결정 요청
                    if predicted_action == 'Buy':
                        stats = self.specialist_stats[current_regime]
                        win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 10 else 0.5
                        avg_profit = stats['total_profit'] / stats['wins'] if stats['wins'] > 0 else 1
                        avg_loss = abs(stats['total_loss'] / stats['losses']) if stats['losses'] > 0 else 1
                        
                        investment_fraction = self.risk_control_tower.determine_investment_size(
                            win_rate, avg_profit, avg_loss, confidence, sentiment_score
                        )
                        
                        if investment_fraction > 0:
                            cash_balance = await self.upbit_service.get_balance('KRW') or 0
                            buy_amount_krw = cash_balance * investment_fraction
                            if buy_amount_krw > 5000:
                                await self.execution_engine.create_market_buy_order(symbol, buy_amount_krw)
                            else:
                                print('  - [EXEC] 주문 금액이 최소 기준(5,000 KRW) 미만입니다.')

                    elif predicted_action == 'Sell':
                        coin_ticker = symbol.split('-')[1] # KRW-BTC -> BTC
                        coin_balance = await self.upbit_service.get_balance(coin_ticker)
                        if coin_balance and coin_balance > 0:
                            await self.execution_engine.create_market_sell_order(symbol, coin_balance)
                        else:
                            print(f'  - [EXEC] 매도할 {coin_ticker} 코인이 없습니다.')
                
                print('\n--- 10분 후 다음 유니버스 사이클 시작 ---')
                await asyncio.sleep(600)

            except Exception as e:
                print('[FATAL] 거래 루프 중 치명적 오류 발생:')
                print(traceback.format_exc())
                await asyncio.sleep(60)
async def main_live(): # capital은 초기 자본금이 아닌, 최대 투자 한도를 의미할 수 있음 
    trader = LiveTrader(capital=1000000) 
    await trader.initialize() 
    await trader.run()

if __name__ == '__main__': 
    try: 
        asyncio.run(main_live()) 
    except Exception as e: 
        print('[FATAL] 봇이 최상위 레벨에서 중지되었습니다.') 
        print(traceback.format_exc()) 
        sys.exit(1)