import pandas as pd

def generate_trend_signals(df: pd.DataFrame, fast_ema_period: int = 12, slow_ema_period: int = 26, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> pd.DataFrame:
    """
    EMA 교차와 MACD 모멘텀 필터를 결합하여 추세 추종 신호를 생성합니다.

    Args:
        df (pd.DataFrame): OHLCV 데이터프레임.
        fast_ema_period (int): 단기 EMA 기간.
        slow_ema_period (int): 장기 EMA 기간.
        macd_fast (int): MACD 단기 기간.
        macd_slow (int): MACD 장기 기간.
        macd_signal (int): MACD 시그널 기간.

    Returns:
        pd.DataFrame: 'trend_signal' 컬럼이 추가된 데이터프레임.
                      (1.0: 매수 전환, -1.0: 매도 전환, 0: 유지)
    """
    df_copy = df.copy()

    # EMA 계산
    df_copy['EMA_fast'] = df_copy['close'].ewm(span=fast_ema_period, adjust=False).mean()
    df_copy['EMA_slow'] = df_copy['close'].ewm(span=slow_ema_period, adjust=False).mean()

    # MACD 계산
    macd = df_copy.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal)
    # MACD 히스토그램을 모멘텀의 척도로 사용 (MACD선 - 시그널선)
    df_copy['MACD_hist'] = macd[f'MACDH_{macd_fast}_{macd_slow}_{macd_signal}']

    # 포지션 상태 결정 (1: 강세, -1: 약세, 0: 중립)
    # 강세 조건: 단기 EMA > 장기 EMA AND MACD 히스토그램 > 0 (상승 모멘텀)
    is_bullish = (df_copy['EMA_fast'] > df_copy['EMA_slow']) & (df_copy['MACD_hist'] > 0)
    # 약세 조건: 단기 EMA < 장기 EMA AND MACD 히스토그램 < 0 (하락 모멘텀)
    is_bearish = (df_copy['EMA_fast'] < df_copy['EMA_slow']) & (df_copy['MACD_hist'] < 0)

    df_copy['position'] = 0.0
    df_copy.loc[is_bullish, 'position'] = 1.0
    df_copy.loc[is_bearish, 'position'] = -1.0

    # 포지션 전환 시점을 거래 신호로 변환
    df_copy['trend_signal'] = df_copy['position'].diff()
    df_copy['trend_signal'] = df_copy['trend_signal'].fillna(0)

    return df_copy