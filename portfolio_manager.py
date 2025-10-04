import asyncio
import numpy as np  # Added numpy
from datetime import datetime
from core.exchange import UpbitService
from strategies.breakout_trader import BreakoutTrader
from dl_model_trainer import DLModelTrainer  # For TARGET_COINS
from rl_model_trainer import RLModelTrainer  # Import RLModelTrainer
from rl_environment import TradingEnv  # Import TradingEnv
import scanner
from market_regime_detector import MarketRegimeDetector


class PortfolioManager:
    def __init__(
        self,
        upbit_service: UpbitService,
        allocated_capital: float,
        max_concurrent_trades: int,
    ):
        self.upbit_service = upbit_service
        self.total_capital = allocated_capital  # 이제 할당된 자본을 의미
        self.max_concurrent_trades = max_concurrent_trades
        self.active_trades = {}
        self.ohlcv_cache = {}

        # 딥러닝 모델 로드
        self.dl_trainer = DLModelTrainer()
        self.dl_trainer.load_model()

        # 강화학습 에이전트 로드
        self.rl_trainer = RLModelTrainer()
        self.rl_agent = self.rl_trainer.load_agent()
        if self.rl_agent is None:
            print(
                "경고: RL 에이전트를 로드할 수 없습니다. '--mode train-rl'로 먼저 훈련시켜 주세요."
            )

        # 시장 체제 감지기 초기화
        self.regime_detector = MarketRegimeDetector()

        print(
            f"포트폴리오 매니저(주력 부대) 초기화 완료. 할당 자본: {self.total_capital:,.0f} KRW, 최대 동시 거래: {self.max_concurrent_trades}"
        )

    async def initialize(self):
        # UpbitService 연결은 AICommander에서 처리하므로 여기서는 별도 연결 불필요
        pass

    async def _run_strategy_task(self, strategy_instance):
        """
        주어진 전략 인스턴스를 별도의 비동기 태스크에서 실행합니다.
        """
        try:
            await strategy_instance.run()
        except Exception as e:
            print(f"Error in strategy task for {strategy_instance.ticker}: {e}")
        finally:
            # 태스크 종료 후 active_trades에서 제거
            if strategy_instance.ticker in self.active_trades:
                del self.active_trades[strategy_instance.ticker]
                print(
                    f"Trade for {strategy_instance.ticker} completed/stopped and removed from active trades."
                )

    async def run(self, scan_interval_seconds: int = 300):  # 5분마다 스캔
        """
        포트폴리오 관리 로직을 실행합니다. (상황 적응형 AI 로직)
        """
        print("주력 부대(PortfolioManager) 운영 시작...")
        if self.rl_agent is None or self.dl_trainer.model is None:
            print(
                "AI 모델이 준비되지 않았습니다. RL 에이전트와 DL 모델을 모두 훈련시켜야 합니다."
            )
            return

        while True:
            try:
                # 1. 현재 활성 거래 수 확인
                current_active_trades = len(self.active_trades)
                if current_active_trades >= self.max_concurrent_trades:
                    # print(f"주력 부대: 최대 동시 거래 수({self.max_concurrent_trades})에 도달하여 기존 거래를 모니터링합니다.")
                    await asyncio.sleep(scan_interval_seconds)
                    continue

                print(
                    f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] 주력 부대: 새로운 거래 기회 탐색 중... (활성 거래: {current_active_trades}/{self.max_concurrent_trades})"
                )

                # 2. 시장 체제 감지
                btc_df_daily = await self.upbit_service.fetch_latest_ohlcv(
                    "BTC/KRW", "day", 201
                )
                market_regime = self.regime_detector.get_market_regime(btc_df_daily)

                # 3. DL 모델을 이용한 핫 코인 스캔
                hot_coins = await scanner.find_hot_coin_live(
                    self.upbit_service.exchange, self.dl_trainer, market_regime
                )

                if not hot_coins:
                    print("주력 부대: 현재 DL 모델 기준에 맞는 핫 코인이 없습니다.")
                    await asyncio.sleep(scan_interval_seconds)
                    continue

                dl_selected_ticker = hot_coins[0]

                if dl_selected_ticker in self.active_trades:
                    print(
                        f"주력 부대: {dl_selected_ticker}는 이미 활성 거래 중이므로 건너뜁니다."
                    )
                    await asyncio.sleep(scan_interval_seconds)
                    continue

                # 4. RL 에이전트의 최종 승인
                print(
                    f"주력 부대: DL 모델 선정 코인({dl_selected_ticker})에 대한 RL 에이전트의 최종 승인 확인 중..."
                )
                window_size = 60  # 훈련 시 사용한 window_size와 동일해야 합니다.
                df_1h = await self.upbit_service.fetch_latest_ohlcv(
                    dl_selected_ticker, "1h", limit=window_size + 5
                )  # 여유분 데이터 확보

                if df_1h.empty or len(df_1h) < window_size:
                    print(
                        f"주력 부대: {dl_selected_ticker}에 대한 RL 에이전트 평가용 데이터가 부족합니다 (필요: {window_size}, 현재: {len(df_1h)})."
                    )
                    await asyncio.sleep(scan_interval_seconds)
                    continue

                # 데이터 전처리 (훈련 시와 동일하게)
                df_1h.drop(columns=["regime"], inplace=True, errors="ignore")
                df_1h.dropna(inplace=True)
                df_1h = df_1h.astype(np.float32)

                if len(df_1h) < window_size:
                    print(
                        f"주력 부대: {dl_selected_ticker}의 데이터가 전처리 후 너무 적어 평가할 수 없습니다."
                    )
                    await asyncio.sleep(scan_interval_seconds)
                    continue

                # 예측을 위한 임시 환경 생성
                pred_env = TradingEnv(df=df_1h.tail(window_size))
                observation, _ = pred_env.reset()

                action, _ = self.rl_agent.predict(observation, deterministic=True)

                if action != 1:  # 1: 매수
                    print(
                        f"주력 부대: RL 에이전트가 {dl_selected_ticker}에 대한 매수를 승인하지 않았습니다 (액션: {action})."
                    )
                    await asyncio.sleep(scan_interval_seconds)
                    continue

                print(f"🧠 주력 부대: RL 에이전트가 {dl_selected_ticker} 매수 승인!")

                # 5. 최종 거래 결정 및 실행
                print(
                    f"✅ 주력 부대: 최종 승인에 따라 {dl_selected_ticker} 거래를 시작합니다."
                )

                # 할당된 자본 내에서 거래 자본 계산
                capital_for_trade = self.total_capital / (
                    self.max_concurrent_trades - current_active_trades
                )

                strategy_instance = BreakoutTrader(
                    self.upbit_service,
                    dl_selected_ticker,
                    allocated_capital=capital_for_trade,
                )

                trade_task = asyncio.create_task(
                    self._run_strategy_task(strategy_instance)
                )
                self.active_trades[dl_selected_ticker] = {
                    "task": trade_task,
                    "strategy": strategy_instance,
                    "capital_allocated": capital_for_trade,
                }
                print(
                    f"{type(strategy_instance).__name__} 전략으로 {dl_selected_ticker} 거래 시작. 할당 자본: {capital_for_trade:,.0f} KRW"
                )

            except Exception as e:
                print(f"포트폴리오 매니저 실행 루프 중 에러 발생: {e}")

            await asyncio.sleep(scan_interval_seconds)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    env_path = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("""UPBIT_ACCESS_KEY=YOUR_ACCESS_KEY
UPBIT_SECRET_KEY=YOUR_SECRET_KEY""")
        print(
            f"Created a dummy .env file at {env_path}. Please replace YOUR_ACCESS_KEY and UPBIT_SECRET_KEY with actual values."
        )
    load_dotenv(env_path)

    async def main_async():
        # PortfolioManager 테스트 예시
        # 실제 API 키가 .env 파일에 설정되어 있어야 합니다.
        try:
            manager = PortfolioManager(
                total_capital=1_000_000, max_concurrent_trades=3
            )  # 100만원, 최대 3개 동시 거래
            await manager.initialize()
            await manager.run(scan_interval_seconds=60)  # 1분마다 스캔
        except Exception as e:
            print(f"PortfolioManager test failed: {e}")

    asyncio.run(main_async())
