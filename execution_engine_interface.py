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

    def __init__(self, upbit_service: UpbitService):
        self.upbit_service = upbit_service
        print("✅ Upbit 주문 실행 엔진 활성화.")

    async def create_market_buy_order(self, symbol: str, amount_krw: float):
        print(f"  - [EXEC] 시장가 매수 주문 실행 -> {symbol} / {amount_krw:,.0f} KRW")
        # 실제 주문 로직 (현재는 주석 처리)
        # return await self.upbit_service.create_market_buy_order(symbol, amount_krw)
        return {
            "status": "ok",
            "symbol": symbol,
            "amount": amount_krw,
        }  # 시뮬레이션용 반환

    async def create_market_sell_order(self, symbol: str, quantity: float):
        print(f"  - [EXEC] 시장가 매도 주문 실행 -> {symbol} / {quantity}개")
        # 실제 주문 로직 (현재는 주석 처리)
        # return await self.upbit_service.create_market_sell_order(symbol, quantity)
        return {
            "status": "ok",
            "symbol": symbol,
            "quantity": quantity,
        }  # 시뮬레이션용 반환

    async def liquidate_all_positions(self, holdings: dict):
        print("🚨 [EXEC] 모든 포지션 즉시 청산 실행!")
        results = []
        for symbol, quantity in holdings.items():
            if quantity > 0:
                result = await self.create_market_sell_order(symbol, quantity)
                results.append(result)
        return results
