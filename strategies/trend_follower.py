import pandas as pd
import pandas_ta as ta

def generate_v_recovery_signals(df: pd.DataFrame, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9) -> pd.DataFrame:
    """
    장기 상승 추세 중 발생하는 단기 조정을 매수 기회로 포착하는 'V-회복' 신호를 생성합니다.
    """
    df_copy = df.copy()

    # 1. 필요 지표 계산 (전처리 단계에서 이미 계산됨)
    # RSI의 5일 이동평균 계산. precompute_all_indicators에서 'RSI_14'를 생성하므로, 이를 사용합니다.
    df_copy['RSI_SMA'] = df_copy['RSI_14'].rolling(window=5).mean()
    # MACD 히스토그램은 'MACDh_12_26_9' 컬럼을 사용합니다.
    df_copy['MACD_hist'] = df_copy['MACDh_12_26_9']

    # 2. 매수/매도 조건 정의
    # -- 매수 조건 --
    # 조건 1: 장기 추세는 강세 (EMA_50은 백테스터에서 이미 계산되어 있다고 가정)
    is_long_term_bull = df_copy['close'] > df_copy['EMA_50']
    # 조건 2: 최근 3봉 이내에 RSI가 45 아래로 하락했었음
    rsi_dipped_below_45 = df_copy['RSI_14'].rolling(window=3).min() < 45
    # [REVISED] 조건 3: 현재 RSI가 자신의 5일 이동평균을 상향 돌파
    rsi_crosses_above_sma = (df_copy['RSI_14'] > df_copy['RSI_SMA']) & (df_copy['RSI_14'].shift(1) <= df_copy['RSI_SMA'].shift(1))

    buy_conditions = is_long_term_bull & rsi_dipped_below_45 & rsi_crosses_above_sma

    # -- 매도 조건 --
    # 조건: MACD 히스토그램이 음수로 전환
    sell_conditions = (df_copy['MACD_hist'] < 0) & (df_copy['MACD_hist'].shift(1) >= 0)

    # 3. 신호 생성
    df_copy['signal'] = 0.0
    df_copy.loc[buy_conditions, 'signal'] = 1.0
    df_copy.loc[sell_conditions, 'signal'] = -1.0
    
    return df_copy
