import pandas as pd
import pandas_ta as ta  # noqa: F401

def precompute_regime_indicators(df: pd.DataFrame, adx_len=14, atr_len=14, natr_ma_len=90, ema_fast_len=20, ema_slow_len=50) -> pd.DataFrame:
    """
    백테스팅에 필요한 모든 시장 체제 관련 지표를 사전에 일괄 계산합니다.
    단기(20), 장기(50) EMA를 추가합니다.
    """
    df_copy = df.copy()

    # ADX, ATR 계산
    adx = df_copy.ta.adx(length=adx_len)
    df_copy['ADX'] = adx[f'ADX_{adx_len}']
    atr = df_copy.ta.atr(length=atr_len)
    df_copy['Normalized_ATR'] = (atr / df_copy['close']) * 100
    df_copy['Normalized_ATR_MA'] = df_copy['Normalized_ATR'].rolling(window=natr_ma_len, min_periods=30).mean()
    
    # [REVISED] 단기 및 장기 EMA 계산
    df_copy['EMA_20'] = df_copy['close'].ewm(span=ema_fast_len, adjust=False).mean()
    df_copy['EMA_50'] = df_copy['close'].ewm(span=ema_slow_len, adjust=False).mean()
    
    df_copy.dropna(inplace=True)
    return df_copy


def get_regime_from_indicators(close: float, ema_20: float, ema_50: float, adx: float, normalized_atr: float, natr_ma: float, adx_trend=25, adx_sideways=20) -> str:
    """
    미리 계산된 지표 값들을 기반으로 시장 체제를 결정합니다.
    BULLISH_CONSOLIDATION 체제 정의를 수정합니다.
    """
    # [REVISED] 강세장 속 숨고르기 체제 정의: 장기 상승 & 단기 조정
    if close > ema_50 and close < ema_20:
        return 'BULLISH_CONSOLIDATION'

    # 기존 추세 결정
    if adx > adx_trend:
        trend = 'TREND'
    elif adx < adx_sideways:
        trend = 'SIDEWAYS'
    else:
        trend = 'UNDEFINED'

    # 변동성 결정
    if normalized_atr > natr_ma:
        volatility = 'HIGH_VOL'
    else:
        volatility = 'LOW_VOL'

    if trend == 'UNDEFINED':
        return 'UNDEFINED'

    return f"{trend}_{volatility}"
