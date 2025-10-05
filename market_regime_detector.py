
import pandas as pd
import pandas_ta as ta

def precompute_regime_indicators(df: pd.DataFrame, adx_len=14, atr_len=14, natr_ma_len=90) -> pd.DataFrame:
    """
    백테스팅에 필요한 모든 시장 체제 관련 지표를 사전에 일괄 계산합니다.
    이 함수는 SettingWithCopyWarning을 방지하기 위해 데이터프레임의 복사본에서 작동합니다.

    Args:
        df (pd.DataFrame): OHLCV 데이터프레임.
        adx_len (int): ADX 계산 기간.
        atr_len (int): ATR 계산 기간.
        natr_ma_len (int): Normalized ATR 이동평균 계산 기간.

    Returns:
        pd.DataFrame: 지표가 추가된 데이터프레임.
    """
    # 원본 데이터프레임을 수정하지 않도록 복사본 생성
    df_copy = df.copy()

    # ADX 계산
    adx = df_copy.ta.adx(length=adx_len)
    df_copy['ADX'] = adx[f'ADX_{adx_len}']

    # Normalized ATR 계산
    atr = df_copy.ta.atr(length=atr_len)
    df_copy['Normalized_ATR'] = (atr / df_copy['close']) * 100

    # Normalized ATR의 이동평균 계산
    df_copy['Normalized_ATR_MA'] = df_copy['Normalized_ATR'].rolling(window=natr_ma_len, min_periods=30).mean()
    
    # 계산된 지표에서 발생할 수 있는 NaN 값 제거
    df_copy.dropna(inplace=True)
    
    return df_copy


def get_regime_from_indicators(adx: float, normalized_atr: float, natr_ma: float, adx_trend=25, adx_sideways=20) -> str:
    """
    미리 계산된 지표 값들을 기반으로 시장 체제를 결정합니다.

    Args:
        adx (float): ADX 값.
        normalized_atr (float): Normalized ATR 값.
        natr_ma (float): Normalized ATR의 이동평균 값.
        adx_trend (int): 추세로 판단할 ADX 임계값.
        adx_sideways (int): 횡보로 판단할 ADX 임계값.

    Returns:
        str: 시장 체제 문자열.
    """
    # 추세 결정
    if adx > adx_trend:
        trend = 'TREND'
    elif adx < adx_sideways:
        trend = 'SIDEWAYS'
    else:
        trend = 'UNDEFINED' # 또는 'TRANSITION'

    # 변동성 결정
    if normalized_atr > natr_ma:
        volatility = 'HIGH_VOL'
    else:
        volatility = 'LOW_VOL'

    # 체제 조합
    if trend == 'UNDEFINED':
        return 'UNDEFINED'

    return f"{trend}_{volatility}"
