import pandas as pd
import numpy as np
import pandas_ta as ta


def get_market_regime_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a standardized market regime detection logic to a DataFrame.
    - Bullish: ADX > 25 and EMA_20 > EMA_50
    - Bearish: ADX > 25 and EMA_20 < EMA_50
    - Sideways: Otherwise
    Adds a 'market_regime' column with the results.
    """
    df_copy = df.copy()
    # Ensure required indicators are present
    if 'ADX_14' not in df_copy.columns or 'EMA_20' not in df_copy.columns or 'EMA_50' not in df_copy.columns:
        df_copy = precompute_all_indicators(df_copy)

    conditions = [
        (df_copy['ADX_14'] > 25) & (df_copy['EMA_20'] > df_copy['EMA_50']),
        (df_copy['ADX_14'] > 25) & (df_copy['EMA_20'] < df_copy['EMA_50'])
    ]
    choices = ['Bullish', 'Bearish']
    df_copy['market_regime'] = np.select(conditions, choices, default='Sideways')
    return df_copy

def precompute_all_indicators(df: pd.DataFrame):
    """
    [REFACTORED] pandas_ta를 사용하여 모든 기술적 지표를 일관되게 계산합니다.
    """
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACDh_12_26_9'] = macd['MACDH_12_26_9']
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['BBL_20'] = bbands['BBL_20']
    df['BBM_20'] = bbands['BBM_20']
    df['BBU_20'] = bbands['BBU_20']
    df['BBP_20_2.0'] = (df['close'] - df['BBL_20']) / (df['BBU_20'] - df['BBL_20'])
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ATRr_14'] = atr
    # df['STOCHRSIk_14_14_3_3'] = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)['STOCHRSIk_14_14_3_3']
    ppo = ta.ppo(df['close'], fast=12, slow=26, signal=9)
    df['PPO_12_26_9'] = ppo['PPO_12_26_9']
    df['PPOh_12_26_9'] = ppo['PPOH_12_26_9']
    df['PPOs_12_26_9'] = ppo['PPOS_12_26_9']
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX_14'] = adx['ADX_14']
    df['DMP_14'] = adx['DMP_14']
    df['DMN_14'] = adx['DMN_14']
    
    # NATR은 수동 계산이 필요할 수 있습니다.
    if 'ATRr_14' in df.columns:
        df['NATR_14'] = (df['ATRr_14'] / df['close']) * 100

    # SMA는 pandas 내장 함수 사용
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    df.dropna(inplace=True)
    return df

def get_market_regime(row: pd.Series) -> str:
    if row['SMA_50'] < row['SMA_200']:
        return 'BEARISH'
    return 'NEUTRAL'