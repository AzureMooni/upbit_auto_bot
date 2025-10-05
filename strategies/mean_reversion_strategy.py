
import pandas as pd

def generate_sideways_signals(df: pd.DataFrame, bband_length=20, rsi_length=14, bband_std=2.0):
    """
    Generates trading signals for a mean-reversion strategy in sideways markets.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        bband_length (int): The lookback period for Bollinger Bands.
        rsi_length (int): The lookback period for RSI.
        bband_std (float): The number of standard deviations for Bollinger Bands.

    Returns:
        pd.DataFrame: The original DataFrame with 'signal' and 'confidence' columns added.
    """
    # Calculate Bollinger Bands and get the %B indicator
    bbands = df.ta.bbands(length=bband_length, std=bband_std)
    # Columns are named like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
    bpercent_col = f'BBP_{bband_length}_{bband_std}'
    df['BBP'] = bbands[bpercent_col]

    # Calculate RSI
    df['RSI'] = df.ta.rsi(length=rsi_length)

    # Generate signals based on %B and RSI
    # Buy when oversold: %B is low and RSI is low
    buy_conditions = (df['BBP'] < 0.1) & (df['RSI'] < 30)
    # Sell when overbought: %B is high and RSI is high
    sell_conditions = (df['BBP'] > 0.9) & (df['RSI'] > 70)

    # Create the signal column
    df['signal'] = 0.0
    df.loc[buy_conditions, 'signal'] = 1.0  # Buy signal
    df.loc[sell_conditions, 'signal'] = -1.0 # Sell signal

    # Set confidence for valid signals
    df['confidence'] = 0.0
    df.loc[buy_conditions | sell_conditions, 'confidence'] = 0.8

    return df

# Example Usage:
if __name__ == '__main__':
    # Create a sample DataFrame (replace with your actual data)
    data = {
        'open': [100, 102, 101, 99, 98, 100, 103, 105, 104, 102] * 3,
        'high': [103, 104, 102, 100, 99, 102, 105, 106, 105, 103] * 3,
        'low': [99, 101, 100, 98, 97, 99, 102, 104, 103, 101] * 3,
        'close': [101, 103, 100, 98, 99, 101, 104, 105, 102, 101] * 3,
        'volume': [1000] * 30
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=30))

    signals_df = generate_sideways_signals(df.copy())
    print("DataFrame with Sideways Signals:")
    print(signals_df.tail(10))
