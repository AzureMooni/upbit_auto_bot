import pandas as pd
import numpy as np

def precompute_all_indicators(df: pd.DataFrame):
    """
    [FINAL] 모든 지표를 pandas 기본 함수만으로 직접 계산하여 안정성을 확보합니다.
    """
    # EMA
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema_fast - ema_slow
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

    # Bollinger Bands
    mid_band = df['close'].rolling(window=20).mean()
    std_dev = df['close'].rolling(window=20).std()
    upper_band = mid_band + (std_dev * 2)
    lower_band = mid_band - (std_dev * 2)
    df['BBP_20_2.0'] = (df['close'] - lower_band) / (upper_band - lower_band)
    df['BBB_20_2.0'] = (upper_band - lower_band) / mid_band

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATRr_14'] = (tr.rolling(window=14).mean() / df['close']) * 100

    # Stochastic RSI
    rsi = df['RSI_14']
    stoch_rsi_k = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
    df['STOCHk_14_14_3_3'] = stoch_rsi_k.rolling(3).mean() * 100
    df['STOCHd_14_14_3_3'] = df['STOCHk_14_14_3_3'].rolling(3).mean()

    # PPO (Percentage Price Oscillator)
    ppo_ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ppo_ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['PPO_12_26_9'] = ((ppo_ema_fast - ppo_ema_slow) / ppo_ema_slow) * 100
    df['PPOs_12_26_9'] = df['PPO_12_26_9'].ewm(span=9, adjust=False).mean()
    df['PPOh_12_26_9'] = df['PPO_12_26_9'] - df['PPOs_12_26_9']

    # Macro Indicators
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()

    # ADX (Average Directional Index)
    # Calculate Directional Movement (DM)
    plus_dm = df['high'].diff().where(df['high'].diff() > df['low'].diff(), 0)
    minus_dm = df['low'].diff().where(df['low'].diff() < df['high'].diff(), 0)
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    # Calculate True Range (TR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate Smoothed True Range (ATR)
    atr = tr.ewm(span=14, adjust=False).mean()

    # Calculate Smoothed Directional Movement
    plus_di = (plus_dm.ewm(span=14, adjust=False).mean() / atr) * 100
    minus_di = (minus_dm.ewm(span=14, adjust=False).mean() / atr) * 100

    # Calculate Directional Index (DX)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX_14'] = dx.ewm(span=14, adjust=False).mean()

    # NATR (Normalized Average True Range)
    df['NATR_14'] = (atr / df['close']) * 100
    
    df.dropna(inplace=True)
    return df

def get_market_regime(row: pd.Series) -> str:
    if row['SMA_50'] < row['SMA_200']:
        return 'BEARISH'
    return 'NEUTRAL'