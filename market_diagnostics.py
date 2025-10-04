import numpy as np
from core.exchange import UpbitService
from sentiment_analyzer import SentimentAnalyzer


class MarketDiagnostics:
    """
    시장의 거시 지표(변동성, 투자 심리)를 종합적으로 분석하는 시장 분석 참모 모듈.
    """

    def __init__(self):
        self.upbit_service = UpbitService()
        self.sentiment_analyzer = SentimentAnalyzer()
        print("✅ 시장 분석 참모 활성화.")

    async def get_volatility_index(self, symbol="BTC/USDT", days=30) -> float | None:
        """
        암호화폐 시장의 변동성 지수(CVI)를 계산합니다.
        여기서는 최근 N일간의 일일 로그 수익률의 연율화 표준편차를 사용합니다.

        Args:
            symbol (str): 변동성 계산의 기준이 될 자산. (기본값: BTC/USDT)
            days (int): 변동성 계산에 사용할 기간(일).

        Returns:
            float | None: 연율화된 변동성 지수. 데이터 수집 실패 시 None.
        """
        print(f"  - [Diagnostics] {symbol}의 {days}일 변동성 지수 계산 중...")
        try:
            # ccxt는 USDT 페어를 지원하므로 KRW를 USDT로 변경
            symbol_usdt = symbol.replace("/KRW", "/USDT")
            ohlcv = await self.upbit_service.get_ohlcv(
                symbol_usdt, timeframe="1d", limit=days + 1
            )
            if ohlcv is None or ohlcv.empty:
                return None

            log_returns = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
            annualized_volatility = (
                log_returns.std() * np.sqrt(365) * 100
            )  # 백분율로 표시
            print(
                f"  - [Diagnostics] 계산된 연율화 변동성: {annualized_volatility:.2f}%"
            )
            return annualized_volatility
        except Exception as e:
            print(f"  - [Diagnostics] 변동성 계산 중 오류: {e}")
            return None

    async def get_market_summary(self) -> dict:
        """
        변동성, 시장 심리 등 거시 지표를 종합하여 요약 보고를 생성합니다.

        Returns:
            dict: 변동성, 공포-탐욕 지수, 핵심 내러티브를 포함하는 딕셔너리.
        """
        print("\n- 시장 거시 상황 분석을 시작합니다...")
        volatility = await self.get_volatility_index()
        fear_greed, narrative = self.sentiment_analyzer.get_fear_greed_index("BTC")

        summary = {
            "volatility_index": volatility,
            "fear_greed_index": fear_greed,
            "market_narrative": narrative,
        }
        return summary


if __name__ == "__main__":
    import asyncio

    async def main():
        diagnostics = MarketDiagnostics()
        summary = await diagnostics.get_market_summary()
        print("\n--- 최종 시장 분석 요약 ---")
        print(
            f"  - 변동성 지수: {summary['volatility_index']:.2f}%"
            if summary["volatility_index"] is not None
            else "- 변동성 지수: 계산 실패"
        )
        print(f"  - 공포-탐욕 지수: {summary['fear_greed_index']}")
        print(f"  - 시장 핵심 내러티브: {summary['market_narrative']}")

    asyncio.run(main())
