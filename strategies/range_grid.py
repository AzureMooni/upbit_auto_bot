import time
from strategies.grid_trading import GridTrader
from scanner import classify_market
from core.exchange import UpbitService

class RangeGridTrader(GridTrader):
    def __init__(self, upbit_service: UpbitService, ticker: str, lower_price: float, upper_price: float, grid_count: int, order_amount_krw: float):
        super().__init__(upbit_service, ticker, lower_price, upper_price, grid_count, order_amount_krw)
        self.market_type = ""
        print(f"RangeGridTrader initialized for {self.ticker}. Will only activate in 'ranging' market.")

    def run(self, interval_seconds: int = 5):
        """
        박스권 그리드 트레이딩 전략을 실행합니다.
        'ranging' 시장에서만 작동합니다.
        """
        print(f"Starting RangeGridTrader for {self.ticker}...")
        while True:
            try:
                # 시장 분류
                self.market_type = classify_market(self.ticker, exchange=self.upbit_service.exchange) # UpbitService의 exchange 객체 전달
                print(f"[{time.strftime('%Y-%m-%d %H:%M')}] Current market type for {self.ticker}: {self.market_type}")

                if self.market_type == 'ranging':
                    print(f"Market is ranging. Activating GridTrader logic for {self.ticker}...")
                    # GridTrader의 run 로직을 직접 호출하거나, 필요한 부분만 복사하여 사용
                    # 여기서는 GridTrader의 run 로직을 재사용하기 위해, 부모 클래스의 run 메서드를 호출하는 방식으로 구현
                    # 하지만 GridTrader의 run은 무한 루프이므로, 직접 호출하면 안됩니다.
                    # 대신, GridTrader의 핵심 로직을 여기에 다시 구현하거나, GridTrader의 run을 한 스텝만 실행하는 메서드로 분리해야 합니다.
                    # 현재 GridTrader의 run은 무한 루프이므로, 여기서는 GridTrader의 핵심 로직을 직접 구현합니다.

                    current_price = self.upbit_service.get_current_price(self.ticker)
                    if current_price is None:
                        print(f"Could not fetch current price for {self.ticker}. Retrying...")
                        time.sleep(interval_seconds)
                        continue

                    print(f"Current price for {self.ticker}: {current_price}")

                    # Stop-loss check (GridTrader의 로직 재사용)
                    if current_price <= self.stop_loss_price:
                        print(f"🚨 손절매 발동! {self.ticker} 전량 시장가 매도 및 거래 중지.")
                        self.upbit_service.cancel_all_orders(self.ticker)
                        base_currency = self.ticker.split('/')[0]
                        balances = self.upbit_service.get_balance()
                        amount_to_sell = balances['coins'].get(base_currency, 0)
                        if amount_to_sell > 0:
                            self.upbit_service.create_market_sell_order(self.ticker, amount_to_sell)
                        else:
                            print(f"Warning: No {base_currency} to sell for stop-loss.")
                        return # 프로그램 종료

                    # 매수 그리드 확인 (GridTrader의 로직 재사용)
                    for grid_price in self.grids:
                        if current_price <= grid_price and self.active_orders.get(grid_price) != 'buy':
                            print(f"Price {current_price} crossed BUY grid line at {grid_price}. Placing BUY order...")
                            order = self._place_order('buy', grid_price)
                            if order:
                                self.active_orders[grid_price] = 'buy'
                            break

                    # 매도 그리드 확인 (GridTrader의 로직 재사용)
                    for grid_price in self.grids:
                        if current_price >= grid_price and self.active_orders.get(grid_price) != 'sell':
                            print(f"Price {current_price} crossed SELL grid line at {grid_price}. Placing SELL order...")
                            order = self._place_order('sell', grid_price)
                            if order:
                                self.active_orders[grid_price] = 'sell'
                            break

                else:
                    print(f"Market is not ranging ({self.market_type}). Waiting for ranging market...")

            except Exception as e:
                print(f"An error occurred in RangeGridTrader run loop: {e}")
            
            time.sleep(interval_seconds)

if __name__ == '__main__':
    # 이 부분은 실제 UpbitService 인스턴스와 연동하여 테스트해야 합니다.
    try:
        upbit_service = UpbitService()
        # upbit_service.connect() # 실제 연결은 main.py에서 수행

        ticker = 'BTC/KRW'
        lower_price = 30000000.0
        upper_price = 40000000.0
        grid_count = 5
        order_amount_krw = 10000.0

        range_grid_trader = RangeGridTrader(upbit_service, ticker, lower_price, upper_price, grid_count, order_amount_krw)
        
        print("RangeGridTrader example setup complete. To run, integrate into main.py and ensure API keys are set.")
        # range_grid_trader.run(interval_seconds=10) # 실제 실행 시 주석 해제

    except Exception as e:
        print(f"An unexpected error occurred during RangeGridTrader setup: {e}")
