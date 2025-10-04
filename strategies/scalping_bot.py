import asyncio
import pandas as pd
from core.exchange import UpbitService

import time


class ScalpingBot:
    def __init__(
        self,
        upbit_service: UpbitService,
        ticker: str,
        allocated_capital: float,
        trade_amount: float,
    ):
        self.upbit_service = upbit_service
        self.ticker = ticker
        self.allocated_capital = allocated_capital
        self.trade_amount = trade_amount
        self.position_held = False
        self.entry_price = 0.0
        self.purchased_qty = 0.0
        self.base_currency = ticker.split("/")[0]
        self.take_profit_ratio = 1.02  # +2% 익절
        self.stop_loss_ratio = 0.99  # -1% 손절

        print(
            f"단기 부대(ScalpingBot) 초기화: {self.ticker}, 할당 자본: {self.allocated_capital:,.0f} KRW, 거래당 금액: {self.trade_amount:,.0f} KRW"
        )

    async def _get_ohlcv(self, timeframe="15m", limit=20):
        try:
            ohlcv = await self.upbit_service.exchange.fetch_ohlcv(
                self.ticker, timeframe, limit=limit
            )
            if not ohlcv or len(ohlcv) < limit:
                return None
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {self.ticker}: {e}")
            return None

    async def run(self, interval_seconds: int = 15):
        print(f"단기 부대(ScalpingBot) 운영 시작: {self.ticker}...")
        while True:
            try:
                df = await self._get_ohlcv(timeframe="15m", limit=20)
                if df is None:
                    await asyncio.sleep(interval_seconds)
                    continue

                df.ta.ema(length=5, append=True, close="close")
                df.ta.ema(length=10, append=True, close="close")

                ema5 = df["EMA_5"].iloc[-1]
                ema10 = df["EMA_10"].iloc[-1]
                prev_ema5 = df["EMA_5"].iloc[-2]
                prev_ema10 = df["EMA_10"].iloc[-2]

                current_price = await self.upbit_service.get_current_price(self.ticker)
                if current_price is None:
                    await asyncio.sleep(interval_seconds)
                    continue

                if not self.position_held:
                    if ema5 > ema10 and prev_ema5 <= prev_ema10:
                        print(
                            f"[{time.strftime('%Y-%m-%d %H:%M')}] 📈 단기 부대: {self.ticker} 골든 크로스 발견. 매수 시도."
                        )
                        order = await self.upbit_service.create_market_buy_order(
                            self.ticker, self.trade_amount
                        )
                        if order and order.get("status") == "closed":
                            self.position_held = True
                            self.entry_price = order.get("average", current_price)
                            self.purchased_qty = order.get(
                                "filled", self.trade_amount / self.entry_price
                            )
                            print(
                                f"🟢 단기 부대: 매수 체결. 수량: {self.purchased_qty}, 가격: {self.entry_price:,.2f} KRW."
                            )
                        else:
                            print("❌ 단기 부대: 매수 주문 실패 또는 미체결.")
                else:
                    take_profit_price = self.entry_price * self.take_profit_ratio
                    stop_loss_price = self.entry_price * self.stop_loss_ratio

                    if (
                        current_price >= take_profit_price
                        or current_price <= stop_loss_price
                    ):
                        reason = (
                            "익절" if current_price >= take_profit_price else "손절"
                        )
                        print(
                            f"[{time.strftime('%Y-%m-%d %H:%M')}] 🎯 단기 부대: {self.ticker} {reason} 조건 도달. 매도 시도."
                        )
                        if self.purchased_qty > 0:
                            order = await self.upbit_service.create_market_sell_order(
                                self.ticker, self.purchased_qty
                            )
                            if order and order.get("status") == "closed":
                                print(f"🔴 단기 부대: 매도 체결 ({reason}).")
                                self.position_held = False
                                self.entry_price = 0.0
                                self.purchased_qty = 0.0
                            else:
                                print("❌ 단기 부대: 매도 주문 실패 또는 미체결.")
                        else:
                            print("단기 부대: 경고 - 매도할 수량이 없습니다.")
                            self.position_held = False  # 상태 초기화

            except Exception as e:
                print(f"단기 부대({self.ticker}) 실행 루프 중 오류: {e}")

            await asyncio.sleep(interval_seconds)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import asyncio

    env_path = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("""UPBIT_ACCESS_KEY=YOUR_ACCESS_KEY
UPBIT_SECRET_KEY=YOUR_SECRET_KEY""")
        print(
            f"Created a dummy .env file at {env_path}. Please replace YOUR_ACCESS_KEY and YOUR_SECRET_KEY with actual values."
        )
    load_dotenv(env_path)

    async def test_scalping_bot():
        # ScalpingBot 테스트 예시
        try:
            upbit_service = UpbitService()
            await upbit_service.connect()

            # 예시 파라미터
            ticker = "BTC/KRW"
            order_amount_krw = 100000  # 10만원

            ScalpingBot(upbit_service, ticker, order_amount_krw)
            # await scalping_bot.run(interval_seconds=15) # 실제 실행 시 주석 해제
            print(
                "ScalpingBot example setup complete. To run, uncomment 'scalping_bot.run()' and ensure API keys are set."
            )

        except Exception as e:
            print(f"An unexpected error occurred during ScalpingBot setup: {e}")

    asyncio.run(test_scalping_bot())
