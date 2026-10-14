from abc import ABC, abstractmethod
from core.exchange import UpbitService


class ExecutionEngineInterface(ABC):
    """
    주문 실행 엔진에 대한 추상 인터페이스입니다.
    향후 C++ 또는 Rust와 같은 저지연 언어로 구현될 수 있는 실행 모듈의 기틀을 마련합니다.
    """

    @abstractmethod
    async def create_market_buy_order(self, symbol: str, amount_krw: float):
        """시장가 매수 주문을 생성합니다."""
        pass

    @abstractmethod
    async def create_market_sell_order(self, symbol: str, quantity: float):
        """시장가 매도 주문을 생성합니다."""
        pass

    @abstractmethod
    async def liquidate_all_positions(self, holdings: dict):
        """보유 중인 모든 포지션을 즉시 청산합니다."""
        pass


class UpbitExecutionEngine(ExecutionEngineInterface):
    """
    Upbit API를 사용하는 구체적인 주문 실행 엔진 구현체입니다.
    """

    def __init__(self, upbit_service: UpbitService, open_positions: dict):
        self.upbit_service = upbit_service
        self.open_positions = open_positions # Reference to the LiveTrader's open_positions
        print("✅ Upbit 주문 실행 엔진 활성화.")

    async def create_market_buy_order(self, symbol: str, amount_krw: float):
        import os
        live_mode = os.getenv("LIVE_TRADING", "true").lower() in ("true", "1")
        print(f"  - [EXEC] 시장가 매수 주문 ({'LIVE' if live_mode else 'SIM'}) -> {symbol} / {amount_krw:,.0f} KRW")
        
        if live_mode:
            try:
                order = await self.upbit_service.create_market_buy_order(symbol, amount_krw)
                if order:
                    price = order.get('average') or await self.upbit_service.get_current_price(symbol)
                    filled_amount = order.get('filled') or (amount_krw / price if price else 0)
                    self.open_positions[symbol] = {'entry_price': price, 'quantity': filled_amount}
                    print(f"  - [EXEC LIVE] {symbol} 매수 완료. 진입 가격: {price}, 수량: {filled_amount}")
                    return order
            except Exception as e:
                print(f"  - [EXEC ERROR] 실제 매수 주문 실패: {e}")

        # 시뮬레이션용 반환 (Fallback or Simulation Mode)
        simulated_price = await self.upbit_service.get_current_price(symbol) or 1.0
        simulated_quantity = amount_krw / simulated_price
        self.open_positions[symbol] = {'entry_price': simulated_price, 'quantity': simulated_quantity}
        print(f"  - [EXEC SIM] {symbol} 매수 시뮬레이션 기록. 진입 가격: {simulated_price}, 수량: {simulated_quantity}")
        return {
            "status": "ok",
            "symbol": symbol,
            "amount": amount_krw,
            "price": simulated_price,
            "quantity": simulated_quantity
        }

    async def create_market_sell_order(self, symbol: str, quantity: float):
        import os
        live_mode = os.getenv("LIVE_TRADING", "true").lower() in ("true", "1")
        print(f"  - [EXEC] 시장가 매도 주문 ({'LIVE' if live_mode else 'SIM'}) -> {symbol} / {quantity}개")
        
        if live_mode:
            try:
                order = await self.upbit_service.create_market_sell_order(symbol, quantity)
                if order:
                    if symbol in self.open_positions:
                        del self.open_positions[symbol]
                    print(f"  - [EXEC LIVE] {symbol} 매도 완료.")
                    return order
            except Exception as e:
                print(f"  - [EXEC ERROR] 실제 매도 주문 실패: {e}")

        # 시뮬레이션용 반환
        if symbol in self.open_positions:
            del self.open_positions[symbol]
        print(f"  - [EXEC SIM] {symbol} 매도 시뮬레이션 기록.")
        return {
            "status": "ok",
            "symbol": symbol,
            "quantity": quantity,
        }

    async def liquidate_all_positions(self, holdings: dict):
        print("🚨 [EXEC] 모든 포지션 즉시 청산 실행!")
        results = []
        for symbol, quantity in holdings.items():
            if quantity > 0:
                result = await self.create_market_sell_order(symbol, quantity)
                results.append(result)
        return results
