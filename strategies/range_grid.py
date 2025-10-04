import asyncio
from strategies.grid_trading import GridTrader
from scanner import classify_market_live  # Changed to classify_market_live
from core.exchange import UpbitService


class RangeGridTrader(GridTrader):
    def __init__(
        self,
        upbit_service: UpbitService,
        ticker: str,
        lower_price: float,
        upper_price: float,
        grid_count: int,
        allocated_capital: float,
    ):
        super().__init__(
            upbit_service,
            ticker,
            lower_price,
            upper_price,
            grid_count,
            allocated_capital,
        )
        self.market_type = ""
        print(
            f"RangeGridTrader initialized for {self.ticker}. Will only activate in 'ranging' market."
        )

    async def run(self, interval_seconds: int = 5):
        """
        박스권 그리드 트레이딩 전략을 실행합니다.
        'ranging' 시장에서만 작동합니다.
        """
        print(f"Starting RangeGridTrader for {self.ticker}...")
        while True:
            try:
                # 시장 분류
                self.market_type = await classify_market_live(
                    self.upbit_service.exchange, self.ticker
                )  # Await call
                print(
                    f"[{asyncio.current_task()._coro.cr_frame.f_globals['time'].strftime('%Y-%m-%d %H:%M')}] Current market type for {self.ticker}: {self.market_type}"
                )

                if self.market_type == "ranging":
                    print(
                        f"Market is ranging. Activating GridTrader logic for {self.ticker}..."
                    )

                    current_price = await self.upbit_service.get_current_price(
                        self.ticker
                    )
                    if current_price is None:
                        print(
                            f"Could not fetch current price for {self.ticker}. Retrying..."
                        )
                        await asyncio.sleep(interval_seconds)
                        continue

                    print(f"Current price for {self.ticker}: {current_price}")

                    # Stop-loss check (GridTrader의 로직 재사용)
                    if current_price <= self.stop_loss_price:
                        print(
                            f"🚨 손절매 발동! {self.ticker} 전량 시장가 매도 및 거래 중지."
                        )
                        await self.upbit_service.cancel_all_orders(self.ticker)
                        base_currency = self.ticker.split("/")[0]
                        balances = await self.upbit_service.get_balance()
                        amount_to_sell = balances["coins"].get(base_currency, 0)
                        if amount_to_sell > 0:
                            await self.upbit_service.create_market_sell_order(
                                self.ticker, amount_to_sell
                            )
                        else:
                            print(f"Warning: No {base_currency} to sell for stop-loss.")
                        return  # 프로그램 종료

                    # 매수 그리드 확인 (GridTrader의 로직 재사용)
                    for grid_price in self.grids:
                        if (
                            current_price <= grid_price
                            and self.active_orders.get(grid_price) != "buy"
                        ):
                            print(
                                f"Price {current_price} crossed BUY grid line at {grid_price}. Placing BUY order..."
                            )
                            order = await self._place_order("buy", grid_price)
                            if order:
                                self.active_orders[grid_price] = "buy"
                            break

                    # 매도 그리드 확인 (GridTrader의 로직 재사용)
                    for grid_price in self.grids:
                        if (
                            current_price >= grid_price
                            and self.active_orders.get(grid_price) != "sell"
                        ):
                            print(
                                f"Price {current_price} crossed SELL grid line at {grid_price}. Placing SELL order..."
                            )
                            order = await self._place_order("sell", grid_price)
                            if order:
                                self.active_orders[grid_price] = "sell"
                            break

                else:
                    print(
                        f"Market is not ranging ({self.market_type}). Waiting for ranging market..."
                    )

            except Exception as e:
                print(f"An error occurred in RangeGridTrader run loop: {e}")

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

    async def test_range_grid_trader():
        # 이 부분은 실제 UpbitService 인스턴스와 연동하여 테스트해야 합니다.
        try:
            upbit_service = UpbitService()
            await upbit_service.connect()

            ticker = "BTC/KRW"
            lower_price = 30000000.0
            upper_price = 40000000.0
            grid_count = 5
            order_amount_krw = 10000.0

            RangeGridTrader(
                upbit_service,
                ticker,
                lower_price,
                upper_price,
                grid_count,
                order_amount_krw,
            )

            print(
                "RangeGridTrader example setup complete. To run, integrate into main.py and ensure API keys are set."
            )
            # await range_grid_trader.run(interval_seconds=10) # 실제 실행 시 주석 해제

        except Exception as e:
            print(f"An unexpected error occurred during RangeGridTrader setup: {e}")

    asyncio.run(test_range_grid_trader())
