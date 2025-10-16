import pandas as pd
import pandas_ta as ta

def precompute_regime_indicators(df: pd.DataFrame, adx_len=14, atr_len=14, natr_ma_len=90, ema_fast_len=20, ema_slow_len=50, sma_slow_len=200):
    """
    백테스팅에 필요한 모든 시장 체제 관련 지표를 사전에 일괄 계산합니다.
    [NEW] 50일, 200일 SMA를 추가합니다.
    """
    df_copy = df.copy()

    # 기존 지표 계산
    df_copy['ADX'] = df_copy.ta.adx(length=adx_len)[f'ADX_{adx_len}']
    atr = df_copy.ta.atr(length=atr_len)
    df_copy['Normalized_ATR'] = (atr / df_copy['close']) * 100
    df_copy['Normalized_ATR_MA'] = df_copy['Normalized_ATR'].rolling(window=natr_ma_len, min_periods=30).mean()
    df_copy['EMA_20'] = df_copy['close'].ewm(span=ema_fast_len, adjust=False).mean()
    
    # [NEW] 50일, 200일 SMA 계산
    df_copy['SMA_50'] = df_copy['close'].rolling(window=ema_slow_len).mean()
    df_copy['SMA_200'] = df_copy['close'].rolling(window=sma_slow_len).mean()
    
    df_copy.dropna(inplace=True)
    return df_copy

def get_market_regime(row: pd.Series, adx_trend=25, adx_sideways=20) -> str:
    """
    미리 계산된 지표 값(DataFrame의 한 행)을 기반으로 시장 체제를 결정합니다.
    'BEARISH' 체제를 최우선으로 판별합니다.
    """
    # [NEW] 1순위: 하락장 방어 프로토콜 (데드 크로스)
    if row['SMA_50'] < row['SMA_200']:
        return 'BEARISH'

    # 2순위: 강세장 속 숨고르기
    if row['close'] > row['SMA_50'] and 20 <= row['ADX'] < 25:
        return 'BULLISH_CONSOLIDATION'

    # 3순위: 기존 추세/횡보 구분
    if row['ADX'] > adx_trend:
        trend = 'TREND'
    elif row['ADX'] < adx_sideways:
        trend = 'SIDEWAYS'
    else:
        trend = 'UNDEFINED'

    # 변동성 결정
    if row['Normalized_ATR'] > row['Normalized_ATR_MA']:
        volatility = 'HIGH_VOL'
    else:
        volatility = 'LOW_VOL'

    if trend == 'UNDEFINED':
        return 'UNDEFINED'

    return f"{trend}_{volatility}"