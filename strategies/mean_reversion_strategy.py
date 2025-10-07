
import pandas as pd

def generate_sideways_signals(df: pd.DataFrame, bband_length=20, rsi_length=14, bband_std=2.0):
    """
    Generates trading signals for a mean-reversion strategy in sideways markets.
    This version manually calculates indicators to avoid pandas-ta accessor issues.
    """
    df_copy = df.copy()

    # --- Manual Indicator Calculations ---
    # 1. Bollinger Bands
    mid_band = df_copy['close'].rolling(window=bband_length).mean()
    std_dev = df_copy['close'].rolling(window=bband_length).std()
    upper_band = mid_band + (std_dev * bband_std)
    lower_band = mid_band - (std_dev * bband_std)
    
    # Calculate %B (BBP)
    df_copy['BBP'] = (df_copy['close'] - lower_band) / (upper_band - lower_band)

    # 2. RSI
    delta = df_copy['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_length).mean()
    rs = gain / loss
    df_copy['RSI_14'] = 100 - (100 / (1 + rs))
    # --- End of Manual Calculations ---

    # Generate signals based on %B and RSI
    buy_conditions = (df_copy['BBP'] < 0.1) & (df_copy['RSI_14'] < 30)
    sell_conditions = (df_copy['BBP'] > 0.9) & (df_copy['RSI_14'] > 70)

    # Create the signal column
    df_copy['signal'] = 0.0
    df_copy.loc[buy_conditions, 'signal'] = 1.0
    df_copy.loc[sell_conditions, 'signal'] = -1.0

    # Set confidence for valid signals
    df_copy['confidence'] = 0.0
    df_copy.loc[buy_conditions | sell_conditions, 'confidence'] = 0.8

    return df_copy
