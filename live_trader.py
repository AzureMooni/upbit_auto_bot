import sys, os, asyncio, pandas as pd, numpy as np, traceback, json
from dotenv import load_dotenv
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation

STOP_LOSS_PCT = 0.05 # 5% 손절매 비율

print("DEBUG: live_trader.py started") # Added for debugging

# --- Load .env file ---
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", ".env"))

# --- Core Module Imports ---
try:
    from universe_manager import get_top_10_coins
    from constants import MODEL_SAVE_PATH, LOOKBACK_WINDOW
    from trading_env_simple import SimpleTradingEnv
    from sentiment_analyzer import SentimentAnalyzer
    from core.exchange import UpbitService
    from market_regime_detector import precompute_all_indicators, get_market_regime, get_market_regime_dataframe
    from risk_control_tower import RiskControlTower
    from execution_engine_interface import UpbitExecutionEngine
except ImportError as e:
    print(f'[FATAL] Failed to import core modules: {e}')
    print(traceback.format_exc())
    sys.exit(1)

# --- 1. Load API Keys ---
class LiveTrader:
    def __init__(self, capital: float):
        # --- 1. Load API Keys ---
        if len(sys.argv) == 3:
            access_key = sys.argv[1]
            secret_key = sys.argv[2]
            print(f'[INFO] API Keys loaded from command-line arguments.')
        elif os.environ.get('UPBIT_ACCESS_KEY') and os.environ.get('UPBIT_SECRET_KEY'):
            access_key = os.environ.get('UPBIT_ACCESS_KEY')
            secret_key = os.environ.get('UPBIT_SECRET_KEY')
            print(f'[INFO] API Keys loaded from environment variables.')
        else:
            print('[FATAL] API Keys were not provided either as command-line arguments or as environment variables.')
            print('Usage: python live_trader.py <ACCESS_KEY> <SECRET_KEY>')
            sys.exit(1)
        print(f'[INFO] API Keys loaded successfully. Access Key starts with: {access_key[:4]}...')

        self.initial_capital = capital
        self.agents = {}
        self.upbit_service = UpbitService(access_key, secret_key)
        self.risk_control_tower = RiskControlTower(mdd_threshold=-0.15)
        self.portfolio_history = pd.Series(dtype=float)
        self.sentiment_analyzer = None
        self.open_positions = {}
        self.execution_engine = UpbitExecutionEngine(self.upbit_service, self.open_positions)

    async def initialize(self):
        print('🚀 AI 퀀트 펀드 시스템 초기화를 시작합니다...')
        await self.upbit_service.connect()
        self._load_agents()
        self.specialist_stats = self._load_specialist_stats()
        self._init_analyzer()
        initial_net_worth = await self.get_total_balance()
        self.portfolio_history[pd.Timestamp.now()] = initial_net_worth
        print('✅ 시스템 초기화 완료.')

    def _load_agents(self):
        print('\n- 훈련된 전문가 AI 에이전트들을 로드합니다...')
        MODEL_SAVE_PATH_BASE = "specialist_agent_"
        regimes = ['Bullish', 'Bearish', 'Sideways']
        for regime in regimes:
            model_path = f"{MODEL_SAVE_PATH_BASE}{regime.lower()}.zip"
            if not os.path.exists(model_path):
                print(f'[FATAL] 치명적 오류: {regime} 모델 파일({model_path})이 없습니다.')
                print('Docker 빌드 과정(build-time training)이 실패했거나 모델이 훈련되지 않았습니다.')
                raise Exception(f'{regime} Model file not found: {model_path}')
            
            print(f'  - [{regime}] {model_path} 로드 시도...')
            self.agents[regime] = PPO.load(model_path)
        print(f'  - 모든 전문가 AI 모델({regimes})을 성공적으로 로드했습니다.')


    def _init_analyzer(self):
        print('\n- Gemini 정보 분석가를 준비합니다...')
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            print('  - 정보 분석가 준비 완료.')
        except Exception as e:
            print(f'  - 경고: {e} (Gemini API 키가 없거나 SentimentAnalyzer 모듈 오류.)')

    def _load_specialist_stats(self):
        stats_file = 'specialist_stats.json'
        print(f'\n- 과거 전문가 AI 성과({stats_file})를 로드합니다...')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                print('  - 성과 데이터 로드 완료.')
                return stats
        else:
            print(f'[WARN] 성과 데이터 파일({stats_file})이 없어 새로 생성합니다.')
            stats = {
                regime: {'wins': 0, 'losses': 0, 'total_profit': 0.0, 'total_loss': 0.0, 'trades': 0}
                for regime in ['Bullish', 'Bearish', 'Sideways']
            }
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            print('  - 기본 성과 데이터 파일 생성 완료.')
            return stats

    async def get_total_balance(self) -> float:
        krw_balance = await self.upbit_service.get_balance('KRW') or 0
        total_asset_value = krw_balance
        all_balances = await self.upbit_service.get_all_balances()
        if not all_balances:
            return krw_balance

        for ticker, balance_info in all_balances.items():
            if ticker == 'KRW': # Skip KRW as it's the base currency
                continue
            if balance_info['balance'] > 0:
                market_ticker = f'{ticker}/KRW' # Changed to BASE/QUOTE format
                current_price = await self.upbit_service.get_current_price(market_ticker)
                if current_price:
                    total_asset_value += balance_info['balance'] * current_price
        return total_asset_value

    async def run(self):
        print('\n-- 🚀 AI 퀀트 펀드 실시간 운영 시작 --')
        while True:
            try:
                # 1. 포트폴리오 상태 업데이트 및 서킷 브레이커
                net_worth = await self.get_total_balance()
                self.portfolio_history[pd.Timestamp.now()] = net_worth
                if self.risk_control_tower.check_mdd_circuit_breaker(self.portfolio_history):
                    all_balances = await self.upbit_service.get_all_balances()
                    holdings_to_liquidate = {f'{ticker}/KRW': info['balance'] for ticker, info in all_balances.items() if info['balance'] > 0 and ticker != 'KRW'}
                    await self.execution_engine.liquidate_all_positions(holdings_to_liquidate)
                    print('🚨 모든 거래가 중단되었습니다. 시스템을 종료합니다.')
                    break

                # --- Stop-Loss Check ---
                for symbol, position_info in list(self.open_positions.items()): # Iterate over a copy
                    current_price = await self.upbit_service.get_current_price(symbol)
                    if current_price is None:
                        print(f"  - [STOP-LOSS] {symbol} 현재 가격을 가져올 수 없습니다. 손절매 확인 건너뜀.")
                        continue

                    if self.risk_control_tower.check_stop_loss(current_price, position_info['entry_price'], STOP_LOSS_PCT):
                        print(f"  - [STOP-LOSS] {symbol} 손절매 발동! {position_info['quantity']}개 매도.")
                        await self.execution_engine.create_market_sell_order(symbol, position_info['quantity'])
                # --- End Stop-Loss Check ---

                # 2. 거래 유니버스 결정
                universe = await get_top_10_coins(self.upbit_service)
                
                # 3. 각 자산에 대한 거래 결정
                for symbol in universe:
                    print(f'\n{pd.Timestamp.now()}: [{symbol}] 분석 시작...')
                    
                    # 3a. 시장 분석 및 전문가 AI 선택
                    btc_df = await self.upbit_service.get_ohlcv('BTC/KRW', '1h', 300) # Changed to BTC/KRW
                    if btc_df is None: continue
                    
                    # Standardized regime detection
                    btc_df_with_regime = get_market_regime_dataframe(btc_df)
                    if btc_df_with_regime.empty:
                        print('  - 시장 진단 데이터 부족')
                        continue
                    current_regime = btc_df_with_regime['market_regime'].iloc[-1]
                    
                    agent_to_use = self.agents.get(current_regime)
                    if not agent_to_use:
                        print(f'경고: [{symbol}]을(를) 담당할 AI 에이전트가 없습니다. (Sideways 모델로 대체)')
                        agent_to_use = self.agents.get('Sideways') # Fallback
                        if not agent_to_use:
                           print(f'[ERROR] 대체 모델도 없습니다. 사이클 건너뛰기.')
                           continue
                           
                    print(f'  - 시장 진단: {current_regime}, 담당 전문가: [{current_regime}] Agent')

                    # 3b. 데이터 준비 및 AI 예측
                    target_df = await self.upbit_service.get_ohlcv(symbol, '1h', 300) # symbol is already in BASE/QUOTE format from universe_manager
                    if target_df is None: continue
                    
                    processed_df = precompute_all_indicators(target_df)
                    if len(processed_df) < 50:
                        print('  - 관측 데이터 부족')
                        continue

                    # Standardized regime mapping
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
                            win_rate=win_rate,
                            avg_profit=avg_profit,
                            avg_loss=avg_loss,
                            prediction_confidence=confidence,
                            sentiment_score=sentiment_score
                        )
                        
                        if investment_fraction > 0:
                            cash_balance = await self.upbit_service.get_balance('KRW') or 0
                            buy_amount_krw = cash_balance * investment_fraction
                            if buy_amount_krw > 5000:
                                await self.execution_engine.create_market_buy_order(symbol, buy_amount_krw) # symbol is already in BASE/QUOTE format
                            else:
                                print('  - [EXEC] 주문 금액이 최소 기준(5,000 KRW) 미만입니다.')

                    elif predicted_action == 'Sell':
                        coin_ticker = symbol.split('/')[0] # BTC/KRW -> BTC
                        coin_balance = await self.upbit_service.get_balance(coin_ticker)
                        if coin_balance and coin_balance > 0:
                            await self.execution_engine.create_market_sell_order(symbol, coin_balance) # symbol is already in BASE/QUOTE format
                        else:
                            print(f'  - [EXEC] 매도할 {coin_ticker} 코인이 없습니다.')
                
                print('\n--- 10분 후 다음 유니버스 사이클 시작 ---')
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