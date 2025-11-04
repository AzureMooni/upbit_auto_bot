import pandas as pd
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
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
    df.ta.ppo(fast=12, slow=26, signal=9, append=True)
    df.ta.adx(length=14, append=True)
    
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