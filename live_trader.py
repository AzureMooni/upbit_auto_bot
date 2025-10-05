import asyncio
import joblib
from core.exchange import UpbitService
from preprocessor import DataPreprocessor  # For generate_features

# 고빈도 스캘핑을 위한 타겟 코인 목록
SCALPING_TARGET_COINS = ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW"]


class LiveTrader:
    """
    XGBoost 모델과 기계적 규칙에 기반한 고빈도 스캘핑 거래 실행기.
    """

    def __init__(self, capital: float):
        self.initial_capital = capital
        self.upbit_service = UpbitService()
        self.model = None
        self.scaler = None
        self.target_coins = SCALPING_TARGET_COINS
        self.positions = {
            ticker: False for ticker in self.target_coins
        }  # 코인별 포지션 보유 상태

    def _load_model(
        self, model_path="price_predictor.pkl", scaler_path="price_scaler.pkl"
    ):
        """훈련된 XGBoost 모델과 스케일러를 로드합니다."""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("✅ XGBoost 모델 및 스케일러 로드 완료.")
        except FileNotFoundError:
            print(
                f"오류: 모델 파일('{model_path}') 또는 스케일러 파일('{scaler_path}')을 찾을 수 없습니다."
            )
            print("먼저 모델 훈련을 실행하세요.")
            raise

    async def _get_prediction(self, ticker: str):
        """단일 코인에 대한 예측을 수행합니다."""
        try:
            df = await self.upbit_service.get_ohlcv(ticker, timeframe="1m", limit=100)
            if df is None or df.empty:
                return None

            df_featured = DataPreprocessor.generate_features(df.copy())
            latest_features = df_featured.tail(1)

            features_to_predict = [
                "RSI_14",
                "BBL_20_2.0",
                "BBM_20_2.0",
                "BBU_20_2.0",
                "MACD_12_26_9",
                "MACDh_12_26_9",
                "MACDs_12_26_9",
            ]

            if latest_features[features_to_predict].isnull().values.any():
                return None  # 지표가 NaN이면 예측 불가

            scaled_features = self.scaler.transform(
                latest_features[features_to_predict]
            )
            prediction = self.model.predict(scaled_features)
            return prediction[0]
        except Exception as e:
            print(f"  - {ticker} 예측 중 오류: {e}")
            return None

    async def _manage_position(self, ticker: str, quantity: float, take_profit_price: float, stop_loss_price: float, buy_order_id: str):
        """매수된 포지션에 대한 익절/손절을 OCO 방식으로 관리합니다."""
        print(f"  - [Position] {ticker} 포지션 관리 시작. 수량: {quantity}, 익절가: {take_profit_price:,.0f}, 손절가: {stop_loss_price:,.0f}")

        # 1. 익절 및 손절 지정가 매도 주문 제출
        tp_order = await self.upbit_service.create_limit_sell_order(ticker, quantity, take_profit_price)
        sl_order = await self.upbit_service.create_limit_sell_order(ticker, quantity, stop_loss_price)

        if not tp_order or not sl_order:
            print(f"  - [Error] {ticker} 익절/손절 주문 제출 실패. 포지션 강제 종료.")
            # 주문 실패 시 시장가로 전량 매도하여 포지션 정리
            await self.upbit_service.create_market_sell_order(ticker, quantity)
            self.positions[ticker] = False
            return

        tp_order_id = tp_order['id']
        sl_order_id = sl_order['id']

        print(f"  - [Order] {ticker} 익절 주문 ID: {tp_order_id}, 손절 주문 ID: {sl_order_id}")

        while self.positions[ticker]:
            try:
                # 주문 상태 조회
                tp_status = await self.upbit_service.fetch_order(tp_order_id, ticker)
                sl_status = await self.upbit_service.fetch_order(sl_order_id, ticker)

                if tp_status and tp_status['status'] == 'closed':
                    print(f"  - [SUCCESS] {ticker} 익절 주문 체결! ({tp_status['price']})")
                    # 다른 주문 취소
                    await self.upbit_service.cancel_order(sl_order_id, ticker)
                    break

                if sl_status and sl_status['status'] == 'closed':
                    print(f"  - [FAILURE] {ticker} 손절 주문 체결! ({sl_status['price']})")
                    # 다른 주문 취소
                    await self.upbit_service.cancel_order(tp_order_id, ticker)
                    break

            except Exception as e:
                print(f"  - {ticker} 포지션 관리 중 오류: {e}")
                break
            await asyncio.sleep(0.5)  # 0.5초마다 주문 상태 확인

        self.positions[ticker] = False
        print(f"  - [Position] {ticker} 포지션 종료.")

    async def run(self):
        """고빈도 스캘핑 거래 로직을 실행합니다."""
        self._load_model()
        await self.upbit_service.connect()
        print("🚀 고빈도 퀀트 스캘핑 시스템 가동...")

        while True:
            try:
                # 이미 포지션을 보유한 코인은 예측에서 제외
                coins_to_scan = [
                    ticker for ticker, held in self.positions.items() if not held
                ]
                if not coins_to_scan:
                    await asyncio.sleep(10)  # 모든 코인 포지션 보유 시 10초 대기
                    continue

                # 여러 코인에 대한 예측을 동시에 수행
                prediction_tasks = [
                    self._get_prediction(ticker) for ticker in coins_to_scan
                ]
                predictions = await asyncio.gather(*prediction_tasks)

                for ticker, prediction in zip(coins_to_scan, predictions):
                    if prediction == 1:  # 1: 매수 신호
                        print(
                            f"🔥 [Signal] {ticker}에서 매수 신호 포착! 즉시 거래 실행."
                        )

                        balance = await self.upbit_service.get_balance("KRW")
                        capital_for_trade = balance * 0.5  # 가용 자본의 50% 사용

                        if capital_for_trade < 5000:  # 최소 주문 금액
                            print("  - 경고: 주문 가능 금액이 부족합니다.")
                            continue

                        order = await self.upbit_service.create_market_buy_order(
                            ticker, capital_for_trade
                        )
                        if order and order.get("status") == "closed":
                            entry_price = order.get(
                                "average",
                                await self.upbit_service.get_current_price(ticker),
                            )
                            quantity = order.get(
                                "filled", capital_for_trade / entry_price
                            )
                            self.positions[ticker] = True

                            # Calculate TP/SL prices here
                            take_profit_price = entry_price * 1.005
                            stop_loss_price = entry_price * 0.996

                            asyncio.create_task(
                                self._manage_position(ticker, quantity, take_profit_price, stop_loss_price, order['id'])
                            )

                await asyncio.sleep(60)  # 1분마다 새로운 캔들 확인

            except Exception as e:
                print(f"메인 루프 오류: {e}")
                await asyncio.sleep(60)


if __name__ == "__main__":
    trader = LiveTrader(capital=50000)
    asyncio.run(trader.run())
