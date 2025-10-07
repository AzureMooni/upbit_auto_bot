
import pyupbit
import pandas as pd
import pandas_ta as ta # noqa: F401
from market_regime_detector import precompute_regime_indicators, get_regime_from_indicators
from strategies.trend_follower import generate_trend_signals
from strategies.mean_reversion_strategy import generate_sideways_signals


def run_debug_check():
    """라이브 데이터를 기반으로 모든 지표와 신호를 시간대별로 로깅하여 문제를 진단합니다."""
    print("🚀 Starting Signal Generation Deep Dive...")

    # 1. 실시간 데이터 다운로드 (지난 72시간, 1시간봉)
    try:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=72)
        if df is None:
            print("❌ Failed to fetch data from Upbit.")
            return
        print(f"✅ Successfully fetched {len(df)} hours of data from Upbit.")
    except Exception as e:
        print(f"❌ An error occurred while fetching data: {e}")
        return

    # 2. 모든 지표 및 신호 일괄 계산
    df_indicators = precompute_regime_indicators(df)
    df_indicators = generate_trend_signals(df_indicators)
    
    df_indicators = generate_sideways_signals(df_indicators)
    df_indicators.rename(columns={'signal': 'sideways_signal'}, inplace=True)

    print("\n--- Daily Market Regime Analysis (Last 48 hours) ---")
    # 3. 시장 체제 분석 (지난 2일)
    for i in range(2, 0, -1):
        target_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=i)
        day_data = df_indicators[df_indicators.index.date == target_date.date()]
        if not day_data.empty:
            last_row = day_data.iloc[-1]
            regime = get_regime_from_indicators(
                adx=last_row['ADX'],
                normalized_atr=last_row['Normalized_ATR'],
                natr_ma=last_row['Normalized_ATR_MA']
            )
            print(f"  - {target_date.date()}: {regime}")

    print("\n--- Hourly Signal Generation Log (Last 48 hours) ---")
    # 4. 시간대별 신호 생성 및 로깅
    last_48_hours_df = df_indicators.tail(48)
    
    for timestamp, row in last_48_hours_df.iterrows():
        log_message = (
            f"[{timestamp}] "
            f"Close={row['close']:<8.0f} | "
            f"ADX={row['ADX']:<5.2f} | "
            f"Norm_ATR={row['Normalized_ATR']:<5.2f} | "
            f"EMA_fast={row['EMA_fast']:<8.0f} | "
            f"EMA_slow={row['EMA_slow']:<8.0f} | "
            f"MACD_hist={row['MACD_hist']:<6.2f} | "
            f"BBP={row['BBP']:<5.2f} | "
            f"RSI={row['RSI_14']:<5.2f} | "
            f"=> trend_sig={row['trend_signal']:<4.1f}, sideways_sig={row['sideways_signal']:<4.1f}"
        )
        print(log_message)

if __name__ == "__main__":
    run_debug_check()
