


import pandas as pd

import pandas_ta as ta



def generate_sideways_signals(df: pd.DataFrame, bband_length=20, rsi_length=14, bband_std=2.0):

    """

    Generates trading signals for a mean-reversion strategy in sideways markets.

    This version now uses pandas_ta for consistency.

    """

    df_copy = df.copy()



    # --- pandas_ta Indicator Calculations ---

    df_copy.ta.bbands(length=bband_length, std=bband_std, append=True)

    df_copy.ta.rsi(length=rsi_length, append=True)



    # Ensure columns exist before using them

    bbp_col = f'BBP_{bband_length}_{bband_std}'

    rsi_col = f'RSI_{rsi_length}'



    if bbp_col not in df_copy.columns or rsi_col not in df_copy.columns:

        print(f"[Error] Indicator columns ('{bbp_col}', '{rsi_col}') not found after calculation.")

        return df # Return original df



    # Generate signals based on %B and RSI

    buy_conditions = (df_copy[bbp_col] < 0.1) & (df_copy[rsi_col] < 30)

    sell_conditions = (df_copy[bbp_col] > 0.9) & (df_copy[rsi_col] > 70)



    # Create the signal column

    df_copy['signal'] = 0.0

    df_copy.loc[buy_conditions, 'signal'] = 1.0

    df_copy.loc[sell_conditions, 'signal'] = -1.0



    # Set confidence for valid signals

    df_copy['confidence'] = 0.0

    df_copy.loc[buy_conditions | sell_conditions, 'confidence'] = 0.8



    return df_copy


