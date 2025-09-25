import time
import math
from core.exchange import UpbitService

class GridTrader:
    def __init__(self, upbit_service: UpbitService, ticker: str, lower_price: float, upper_price: float, grid_count: int, order_amount_krw: float):
        self.upbit_service = upbit_service
        self.ticker = ticker
        self.lower_price = lower_price
        self.upper_price = upper_price
        self.grid_count = grid_count
        self.order_amount_krw = order_amount_krw
        self.grids = self._generate_grids()
        self.active_orders = {} # {price: 'buy'/'sell'} to track active grid lines
        print(f"GridTrader initialized for {self.ticker} with {self.grid_count} grids from {self.lower_price} to {self.upper_price}")
        print(f"Generated Grids: {self.grids}")

    def _generate_grids(self):
        """
        그리드 가격 라인을 생성합니다.
        """
        price_range = self.upper_price - self.lower_price
        interval = price_range / (self.grid_count + 1) # 그리드 개수 + 1 만큼의 간격
        
        grids = []
        for i in range(1, self.grid_count + 1):
            grids.append(self.lower_price + i * interval)
        
        # 그리드 가격을 내림차순으로 정렬하여 상단부터 하단으로 확인하기 쉽게 합니다.
        return sorted(grids, reverse=True)

    def _place_order(self, order_type: str, price: float):
        """
        지정가 매수 또는 매도 주문을 실행합니다.
        """
        symbol = self.ticker # 예: 'BTC/KRW'
        base_currency = symbol.split('/')[0] # 예: 'BTC'
        quote_currency = symbol.split('/')[1] # 예: 'KRW'

        if order_type == 'buy':
            # 매수할 코인 수량 계산 (주문 금액 / 현재 가격)
            amount = self.order_amount_krw / price
            try:
                order = self.upbit_service.exchange.create_limit_buy_order(symbol, amount, price)
                print(f"Placed BUY order: {amount:.4f} {base_currency} at {price} {quote_currency}. Order ID: {order['id']}")
                return order
            except Exception as e:
                print(f"Error placing BUY order: {e}")
                return None
        elif order_type == 'sell':
            # 매도할 코인 수량 계산 (주문 금액 / 현재 가격)
            # 실제 보유 수량을 확인해야 하지만, 여기서는 단순화를 위해 order_amount_krw를 사용
            # 실제 구현에서는 get_balance()를 통해 보유 코인 수량을 확인해야 합니다.
            amount = self.order_amount_krw / price # 이 부분은 실제 매도 가능 수량으로 대체되어야 함
            try:
                order = self.upbit_service.exchange.create_limit_sell_order(symbol, amount, price)
                print(f"Placed SELL order: {amount:.4f} {base_currency} at {price} {quote_currency}. Order ID: {order['id']}")
                return order
            except Exception as e:
                print(f"Error placing SELL order: {e}")
                return None
        return None

    def run(self, interval_seconds: int = 5):
        """
        그리드 트레이딩 전략을 실행합니다.
        주기적으로 현재 가격을 확인하고 그리드 라인을 지날 때 주문을 실행합니다.
        """
        print(f"Starting GridTrader for {self.ticker}...")
        while True:
            try:
                current_price = self.upbit_service.get_current_price(self.ticker)
                if current_price is None:
                    print(f"Could not fetch current price for {self.ticker}. Retrying...")
                    time.sleep(interval_seconds)
                    continue

                print(f"Current price for {self.ticker}: {current_price}")

                # 매수 그리드 확인 (가격이 하락하여 그리드 라인에 도달)
                for grid_price in self.grids:
                    if current_price <= grid_price and self.active_orders.get(grid_price) != 'buy':
                        print(f"Price {current_price} crossed BUY grid line at {grid_price}. Placing BUY order...")
                        order = self._place_order('buy', grid_price)
                        if order:
                            self.active_orders[grid_price] = 'buy' # 주문 성공 시 상태 업데이트
                            # TODO: 실제 주문 체결 여부 확인 및 그리드 상태 관리 로직 추가 필요
                            # 예: 체결되면 해당 그리드 라인에서 매도 주문을 걸 준비
                        break # 한 번에 하나의 그리드만 처리

                # 매도 그리드 확인 (가격이 상승하여 그리드 라인에 도달)
                # 이 부분은 매수 주문이 체결된 후 해당 그리드에서 매도 주문을 걸거나,
                # 초기 보유 자산으로 상위 그리드에서 매도 주문을 걸 때 사용됩니다.
                # 여기서는 단순화를 위해 매수 그리드와 별개로 동작하도록 구현합니다.
                for grid_price in self.grids:
                    if current_price >= grid_price and self.active_orders.get(grid_price) != 'sell':
                        print(f"Price {current_price} crossed SELL grid line at {grid_price}. Placing SELL order...")
                        order = self._place_order('sell', grid_price)
                        if order:
                            self.active_orders[grid_price] = 'sell' # 주문 성공 시 상태 업데이트
                            # TODO: 실제 주문 체결 여부 확인 및 그리드 상태 관리 로직 추가 필요
                            # 예: 체결되면 해당 그리드 라인에서 매수 주문을 걸 준비
                        break # 한 번에 하나의 그리드만 처리

            except Exception as e:
                print(f"An error occurred in GridTrader run loop: {e}")
            
            time.sleep(interval_seconds)

if __name__ == '__main__':
    # 이 부분은 실제 UpbitService 인스턴스와 연동하여 테스트해야 합니다.
    # .env 파일에 UPBIT_ACCESS_KEY와 UPBIT_SECRET_KEY가 설정되어 있어야 합니다.
    try:
        # UpbitService 인스턴스 생성 및 연결 (실제 API 키 필요)
        upbit_service = UpbitService()
        upbit_service.connect()

        # GridTrader 인스턴스 생성
        # 예시 파라미터 (실제 사용 시 조정 필요)
        ticker = 'BTC/KRW'
        lower_price = 30000000.0 # 3천만원
        upper_price = 40000000.0 # 4천만원
        grid_count = 5
        order_amount_krw = 10000.0 # 각 그리드 라인에서 1만원씩 주문

        grid_trader = GridTrader(upbit_service, ticker, lower_price, upper_price, grid_count, order_amount_krw)
        
        # 그리드 트레이딩 실행 (예: 10초마다 가격 확인)
        # 이 코드는 무한 루프이므로, 실제 사용 시에는 별도의 스레드나 프로세스로 실행하거나
        # 종료 조건을 추가해야 합니다.
        # grid_trader.run(interval_seconds=10) 
        print("GridTrader example setup complete. To run, uncomment 'grid_trader.run()' and ensure API keys are set.")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except ConnectionError as e:
        print(f"Connection Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during GridTrader setup: {e}")
