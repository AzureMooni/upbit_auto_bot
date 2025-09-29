import pandas as pd

class MarketRegimeDetector:
    """
    비트코인 가격의 이동평균선을 기반으로 현재 시장 체제(regime)를 판단합니다.
    """
    def get_market_regime(self, btc_ohlcv_df: pd.DataFrame) -> str:
        """
        비트코인의 일봉 OHLCV 데이터프레임을 기반으로 시장 체제를 결정합니다.

        Args:
            btc_ohlcv_df (pd.DataFrame): 'close' 컬럼을 포함하는 비트코인의 일봉 데이터.

        Returns:
            str: 'Bullish'(상승장), 'Bearish'(하락장), 또는 'Sideways'(횡보장).
        """
        # 50일, 200일 이동평균선을 계산하기에 데이터가 충분한지 확인
        if btc_ohlcv_df is None or len(btc_ohlcv_df) < 200:
            return 'Sideways'  # 데이터 부족 시 기본값 '횡보장' 반환

        # 50일, 200일 이동평균선 계산
        ma50 = btc_ohlcv_df['close'].rolling(window=50).mean().iloc[-1]
        ma200 = btc_ohlcv_df['close'].rolling(window=200).mean().iloc[-1]

        # 이동평균선 값이 유효한지 확인
        if pd.isna(ma50) or pd.isna(ma200):
            return 'Sideways'  # 이동평균선 계산 불가 시 기본값 반환

        # 골든크로스/데드크로스 조건에 따라 시장 체제 판단
        if ma50 > ma200:
            return 'Bullish'
        elif ma50 < ma200:
            return 'Bearish'
        else:
            return 'Sideways'
