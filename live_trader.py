import sys, os, asyncio, pandas as pd, numpy as np, torch, traceback, json 
from stable_baselines3 import PPO

# --- Core Module Imports ---
# This block fixes all previous ModuleNotFound/ImportErrors
try:
    from universe_manager import get_top_10_coins 
    from trading_env_simple import SimpleTradingEnv 
    from sentiment_analyzer import SentimentAnalyzer 
    from core.exchange import UpbitService 
    from market_regime_detector import precompute_all_indicators, get_market_regime 
    from risk_control_tower import RiskControlTower 
    from execution_engine_interface import UpbitExecutionEngine
    from foundational_model_trainer import MODEL_SAVE_PATH
except ImportError as e:
    print(f'[FATAL] Failed to import core modules: {e}') 
    print(traceback.format_exc()) 
    sys.exit(1)

# --- 1. Load API Keys from Command-Line Arguments ---
if len(sys.argv) != 3:
    print('[FATAL] API Keys were not provided as command-line arguments.') 
    print('Usage: python live_trader.py <ACCESS_KEY> <SECRET_KEY>') 
    sys.exit(1) 
access_key = sys.argv[1] 
secret_key = sys.argv[2] 
print(f'[INFO] API Keys loaded successfully. Access Key starts with: {access_key[:4]}...')

# --- 2. Live Trader Class Definition ---
class LiveTrader:
    def __init__(self, capital: float): 
        self.initial_capital = capital 
        self.agents = {} 
        self.upbit_service = UpbitService(access_key, secret_key)
        self.risk_control_tower = RiskControlTower(mdd_threshold=-0.15)
        self.execution_engine = UpbitExecutionEngine(self.upbit_service)
        self.specialist_stats = self._load_specialist_stats()
        self.portfolio_history = pd.Series(dtype=float)
        self.sentiment_analyzer = None # Initialize as None

    async def initialize(self):
        print('🚀 AI 퀀트 펀드 시스템 초기화를 시작합니다...')
        await self.upbit_service.connect() # This will now find files
        self._load_agents() 
        self._init_analyzer()
        initial_net_worth = await self.get_total_balance()
        self.portfolio_history[pd.Timestamp.now()] = initial_net_worth
        print('✅ 시스템 초기화 완료.')

    def _load_agents(self):
        print("\n- 훈련된 전문가 AI 에이전트들을 로드합니다...")
        model_path = MODEL_SAVE_PATH # e.g., 'foundational_agent.zip'

        if not os.path.exists(model_path):
            print(f'[FATAL] 치명적 오류: 모델 파일({model_path})이 없습니다.')
            print('Docker 빌드 과정(build-time training)이 실패했습니다.')
            raise Exception(f'Model file not found: {model_path}')

        try:
            # 로딩용 더미 환경 생성
            dummy_df = pd.DataFrame(np.random.rand(100, 21), columns=[f'f{i}' for i in range(21)])
            dummy_env = SimpleTradingEnv(dummy_df)
        except Exception as e:
            print(f'[WARN] Dummy env for loading failed: {e}')
            dummy_env = None

        print(f'  - [Foundational] {model_path} 로드 시도...')
        foundational_model = PPO.load(model_path, env=dummy_env)
        
        regimes = ['Bullish', 'Bearish', 'Sideways']
        for regime in regimes:
            self.agents[regime] = foundational_model
        print(f'  - 모든 시장({regimes})에 기본 모델을 성공적으로 할당했습니다.')

    def _init_analyzer(self):
        print('\n- Gemini 정보 분석가를 준비합니다...')
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            print('  - 정보 분석가 준비 완료.')
        except Exception as e:
            print(f'  - 경고: {e} (Gemini API 키가 없어 감성 분석을 건너뜁니다.)')

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
        krw_balance = await self.upbit_service.get_balance('KRW') or 0
        total_asset_value = krw_balance
        all_balances = await self.upbit_service.get_all_balances()
        if not all_balances:
            return krw_balance

        for ticker, balance_info in all_balances.items():
            if balance_info['balance'] > 0:
                market_ticker = f'KRW-{ticker}'
                current_price = await self.upbit_service.get_current_price(market_ticker)
                if current_price:
                    total_asset_value += balance_info['balance'] * current_price
        return total_asset_value

    async def run(self):
        print('\n-- 🚀 AI 퀀트 펀드 실시간 운영 시작 --')
        while True:
            try:
                # ... (Insert full, correct trading loop logic here) ...
                print('--- 10분 후 다음 유니버스 사이클 시작 ---')
                await asyncio.sleep(600)

            except Exception as e:
                print('[FATAL] 거래 루프 중 치명적 오류 발생:')
                print(traceback.format_exc())
                await asyncio.sleep(60)
async def main_live(): 
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
