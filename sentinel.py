import os
import pandas as pd
import pyupbit
import requests  # [NEW] requests 라이브러리 추가
from datetime import datetime

# --- Configuration ---
SYMBOL = "KRW-BTC"
DATA_DIR = "data/retraining_sets"
NTFY_TOPIC = "upbit-sentinel-alerts-a1b2c3d4" # [NEW] ntfy 토픽 설정

def fetch_data():
    """
    진단을 위해 업비트에서 15분봉 및 1시간봉 데이터를 다운로드합니다.
    """
    print("[INFO] Fetching recent market data from Upbit...")
    try:
        # 1. 단기 분석용 15분봉 데이터 (지난 24시간)
        df_15min = pyupbit.get_ohlcv(SYMBOL, interval="minute15", count=96) # 24 * 4
        # 2. 장기 추세 분석용 1시간봉 데이터 (지난 300시간)
        df_1h = pyupbit.get_ohlcv(SYMBOL, interval="minute60", count=300)
        
        if df_15min is None or df_1h is None:
            print("[ERROR] Failed to fetch data from Upbit.")
            return None, None
            
        print(f"[SUCCESS] Fetched {len(df_15min)} 15-min candles and {len(df_1h)} 1-hour candles.")
        return df_15min, df_1h
    except Exception as e:
        print(f"[ERROR] An exception occurred during data fetching: {e}")
        return None, None

def find_missed_opportunities(df_15min: pd.DataFrame, df_1h: pd.DataFrame):
    """
    "V-자 회복" 패턴을 기반으로 놓친 거래 기회를 탐지합니다.
    
    Returns:
        Tuple[pd.DataFrame, pd.Timestamp] | Tuple[None, None]: 기회 데이터와 시점, 또는 None.
    """
    # --- 1. Macro Trend Check (1-hour data) ---
    df_1h['SMA_50'] = df_1h['close'].rolling(window=50).mean()
    df_1h['SMA_200'] = df_1h['close'].rolling(window=200).mean()
    
    if df_1h.iloc[-1]['SMA_50'] <= df_1h.iloc[-1]['SMA_200']:
        print("[INFO] Macro trend is not bullish. Sentinel is standing down.")
        return None, None
    
    print("[INFO] Macro trend is bullish. Scanning for V-Recovery patterns...")

    # --- 2. & 3. Dip and Recovery Check (15-min data) ---
    # 최근 6시간(24봉)을 기준으로 고점을 찾고, 하락 후 회복하는 패턴을 탐지
    lookback_period = 24  # 6 hours
    recovery_period = 8   # 2 hours
    
    # 뒤에서부터 순회하며 패턴을 찾음 (최신 데이터를 먼저 보도록)
    for i in range(len(df_15min) - recovery_period - 1, lookback_period, -1):
        
        # --- Identify a "Dip" ---
        recent_window = df_15min.iloc[i - lookback_period : i]
        peak_price = recent_window['high'].max()
        dip_candle = df_15min.iloc[i]
        
        if dip_candle['low'] <= peak_price * (1 - 0.015): # 1.5% 이상 하락한 지점
            
            # --- Identify a "Recovery" ---
            drop_amount = peak_price - dip_candle['low']
            recovery_target_price = dip_candle['low'] + (drop_amount * 0.5) # 하락분의 50% 회복 목표
            
            recovery_window = df_15min.iloc[i + 1 : i + 1 + recovery_period]
            
            if recovery_window['high'].max() >= recovery_target_price:
                # V-회복 기회 포착 성공!
                opportunity_start_index = recent_window['high'].idxmax()
                opportunity_end_index = recovery_window['high'].idxmax()
                opportunity_segment = df_15min.loc[opportunity_start_index : opportunity_end_index]
                
                return opportunity_segment, dip_candle.name # 기회 구간의 데이터와, 딥이 발생한 시점 반환

    return None, None # 기회 없음

def send_notification(opportunity_timestamp):
    """
    [REVISED] 놓친 기회에 대한 알림을 ntfy.sh를 통해 푸시 알림으로 보냅니다.
    """
    title = "🚨 AI Commander: 놓친 거래 기회 포착!"
    message = f"V-회복 패턴이 {opportunity_timestamp} 경에 발생했으나, 에이전트가 진입하지 않았습니다. 재훈련을 위해 해당 데이터를 저장합니다."
    
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode('utf-8'),
            headers={
                "Title": title.encode('utf-8'),
                "Priority": "high", # 긴급 알림
                "Tags": "warning"    # 경고 아이콘
            }
        )
        print(f"[INFO] Notification sent successfully to ntfy topic: {NTFY_TOPIC}")
    except Exception as e:
        print(f"[ERROR] Failed to send ntfy notification: {e}")

    # 콘솔에도 로그를 남깁니다.
    print("\n" + "="*60)
    print(f"[!!] {title}")
    print(message)
    print("="*60 + "\n")

def trigger_retraining_pipeline(data_segment: pd.DataFrame, timestamp: pd.Timestamp):
    """
    놓친 기회 데이터를 저장하여 재훈련 파이프라인을 시뮬레이션합니다.
    """
    print("[INFO] Initializing automated retraining pipeline...")
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        filename = f"missed_opportunity_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        data_segment.to_csv(filepath)
        
        print(f"[SUCCESS] Saved data for retraining to: {filepath}")
        print("The agent will learn from this in the next training cycle.")
    except Exception as e:
        print(f"[ERROR] Failed to save retraining data: {e}")

def main():
    """
    스크립트의 메인 실행 로직
    """
    df_15min, df_1h = fetch_data()
    
    if df_15min is None or df_1h is None:
        return

    opportunity_segment, opportunity_timestamp = find_missed_opportunities(df_15min, df_1h)
    
    if opportunity_segment is not None:
        send_notification(opportunity_timestamp)
        trigger_retraining_pipeline(opportunity_segment, opportunity_timestamp)
    else:
        print(f"[OK] No missed opportunities detected in the last 24 hours. ({datetime.now().strftime('%Y-%m-%d %H:%M')})")

if __name__ == '__main__':
    main()