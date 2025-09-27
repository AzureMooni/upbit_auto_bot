import time
import threading
from core.exchange import UpbitService
from strategies.breakout_trader import BreakoutTrader # Only BreakoutTrader is used
from scanner import find_hot_coin_live # Only find_hot_coin_live is used

class PortfolioManager:
    def __init__(self, total_capital: float, max_concurrent_trades: int):
        self.total_capital = total_capital
        self.max_concurrent_trades = max_concurrent_trades
        self.upbit_service = UpbitService()
        self.upbit_service.connect()
        self.active_trades = {} # {ticker: {'thread': thread_obj, 'strategy': strategy_obj, 'capital_allocated': float}}
        self.lock = threading.Lock() # 동시성 제어를 위한 락

        print(f"PortfolioManager initialized with total capital: {self.total_capital:,.0f} KRW, max concurrent trades: {self.max_concurrent_trades}")

    def _run_strategy_in_thread(self, strategy_instance):
        """
        주어진 전략 인스턴스를 별도의 스레드에서 실행합니다.
        """
        try:
            strategy_instance.run()
        except Exception as e:
            print(f"Error in strategy thread for {strategy_instance.ticker}: {e}")
        finally:
            # 스레드 종료 후 active_trades에서 제거
            with self.lock:
                if strategy_instance.ticker in self.active_trades:
                    del self.active_trades[strategy_instance.ticker]
                    print(f"Trade for {strategy_instance.ticker} completed/stopped and removed from active trades.")

    def run(self, scan_interval_seconds: int = 300): # 5분마다 스캔
        """
        포트폴리오 관리 로직을 실행합니다.
        """
        print("Starting PortfolioManager run loop (ML-based Breakout Strategy)...")
        while True:
            try:
                # 1. 현재 활성 거래 수 확인
                with self.lock:
                    current_active_trades = len(self.active_trades)
                
                if current_active_trades < self.max_concurrent_trades:
                    # 2. 유망 코인 리스트 가져오기 (ML 모델 기반)
                    print(f"Scanning for hot coins using ML model... (Active trades: {current_active_trades}/{self.max_concurrent_trades})")
                    hot_coins = find_hot_coin_live(self.upbit_service.exchange)
                    
                    if hot_coins:
                        # ML 모델은 가장 높은 확률의 코인 하나만 반환하도록 변경되었으므로, 첫 번째 코인만 사용
                        hot_coin_ticker = hot_coins[0]

                        with self.lock:
                            if hot_coin_ticker in self.active_trades:
                                print(f"Skipping {hot_coin_ticker}: Already an active trade.")
                                # Continue to next iteration of while loop, not next hot coin
                                time.sleep(scan_interval_seconds)
                                continue
                            if len(self.active_trades) >= self.max_concurrent_trades:
                                print("Max concurrent trades reached. Waiting for next scan cycle.")
                                time.sleep(scan_interval_seconds)
                                continue # 최대 동시 거래 개수에 도달하면 스캔 중단

                        print(f"ML model found promising coin: {hot_coin_ticker}. Launching BreakoutTrader.")
                        
                        # 3. 자본 할당
                        current_total_capital = self.upbit_service.get_total_capital()
                        capital_for_trade = current_total_capital / (self.max_concurrent_trades - current_active_trades) if (self.max_concurrent_trades - current_active_trades) > 0 else current_total_capital # 남은 슬롯에 균등 분배
                        
                        strategy_instance = BreakoutTrader(
                            self.upbit_service,
                            hot_coin_ticker,
                            allocated_capital=capital_for_trade
                        )

                        # 별도의 스레드로 전략 실행
                        trade_thread = threading.Thread(target=self._run_strategy_in_thread, args=(strategy_instance,))
                        trade_thread.daemon = True # 메인 프로그램 종료 시 함께 종료
                        trade_thread.start()
                        with self.lock:
                            self.active_trades[hot_coin_ticker] = {
                                'thread': trade_thread,
                                'strategy': strategy_instance,
                                'capital_allocated': capital_for_trade
                            }
                        print(f"Launched {type(strategy_instance).__name__} for {hot_coin_ticker} with {capital_for_trade:,.0f} KRW allocated.")
                    else:
                        print("ML model found no hot coins with high buy probability in this scan cycle. Waiting...")
                else:
                    print("Max concurrent trades reached. Monitoring existing trades.")

            except Exception as e:
                print(f"An error occurred in PortfolioManager run loop: {e}")
            
            time.sleep(scan_interval_seconds)

if __name__ == '__main__':
    # 테스트를 위한 임시 .env 파일 생성 (필요시)
    import os
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("""UPBIT_ACCESS_KEY=YOUR_ACCESS_KEY
UPBIT_SECRET_KEY=YOUR_SECRET_KEY""")
        print(f"Created a dummy .env file at {env_path}. Please replace YOUR_ACCESS_KEY and UPBIT_SECRET_KEY with actual values.")
    load_dotenv(env_path)

    # PortfolioManager 테스트 예시
    # 실제 API 키가 .env 파일에 설정되어 있어야 합니다.
    try:
        manager = PortfolioManager(total_capital=1_000_000, max_concurrent_trades=3) # 100만원, 최대 3개 동시 거래
        manager.run(scan_interval_seconds=60) # 1분마다 스캔
    except Exception as e:
        print(f"PortfolioManager test failed: {e}")
