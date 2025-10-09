
import os
import pandas as pd
import pyupbit
import requests
from datetime import datetime
from universe_manager import get_top_10_coins
from dl_predictor import train_price_prediction_model

# --- Configuration ---
DATA_DIR = "data/retraining_sets"
MODEL_PATH = "data/v2_lstm_model.h5"
NTFY_TOPIC = "upbit-sentinel-v2-alerts"

def find_missed_v_recovery(df_15min: pd.DataFrame, df_1h: pd.DataFrame):
    """ "V-자 회복" 패턴을 기반으로 놓친 거래 기회를 탐지합니다. """
    df_1h['SMA_50'] = df_1h['close'].rolling(window=50).mean()
    df_1h['SMA_200'] = df_1h['close'].rolling(window=200).mean()
    
    if df_1h.empty or df_1h.iloc[-1]['SMA_50'] <= df_1h.iloc[-1]['SMA_200']:
        return None, None
    
    lookback_period, recovery_period = 24, 8
    for i in range(len(df_15min) - recovery_period - 1, lookback_period, -1):
        recent_window = df_15min.iloc[i - lookback_period : i]
        peak_price = recent_window['high'].max()
        dip_candle = df_15min.iloc[i]
        
        if dip_candle['low'] <= peak_price * (1 - 0.015):
            drop_amount = peak_price - dip_candle['low']
            recovery_target_price = dip_candle['low'] + (drop_amount * 0.5)
            recovery_window = df_15min.iloc[i + 1 : i + 1 + recovery_period]
            
            if recovery_window['high'].max() >= recovery_target_price:
                opportunity_start_index = recent_window['high'].idxmax()
                opportunity_end_index = recovery_window['high'].idxmax()
                opportunity_segment = df_15min.loc[opportunity_start_index : opportunity_end_index]
                return opportunity_segment, dip_candle.name
    return None, None

def send_notification(ticker, opportunity_timestamp):
    """ ntfy.sh를 통해 푸시 알림을 보냅니다. """
    title = f"🚨 Sentinel Alert: Missed Opportunity in {ticker}"
    message = f"V-Recovery pattern detected around {opportunity_timestamp}. Triggering auto-retraining pipeline."
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=message.encode('utf-8'), headers={"Title": title.encode('utf-8')})
        print(f"[INFO] Notification sent for {ticker}.")
    except Exception as e:
        print(f"[ERROR] Failed to send notification: {e}")

def trigger_retraining_pipeline(ticker: str, data_segment: pd.DataFrame):
    """
    놓친 기회 데이터를 저장하고, 모델을 재훈련(미세 조정)합니다.
    """
    print(f"[INFO] Initializing automated retraining for {ticker}...")
    try:
        # 1. 재훈련 데이터셋 저장
        os.makedirs(DATA_DIR, exist_ok=True)
        filename = f"missed_{ticker.replace('/','-')}_{datetime.now().strftime('%Y%m%d%H%M')}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        data_segment.to_csv(filepath)
        print(f"[SUCCESS] Saved new training data to: {filepath}")

        # 2. 모델 재훈련 (Fine-tuning)
        # 실제 운영 시에는 전체 데이터를 다시 불러와 합쳐서 훈련해야 함
        # 여기서는 개념 증명을 위해 해당 코인 데이터로만 미세 조정
        print(f"[INFO] Fine-tuning the model with new data from {ticker}...")
        full_data = pyupbit.get_ohlcv(ticker, "minute60", 2000) # 예시 데이터
        # 여기에 새로 찾은 데이터를 합치는 로직 추가
        
        train_price_prediction_model(full_data, MODEL_PATH)
        print(f"[SUCCESS] Model has been retrained and updated at {MODEL_PATH}.")

    except Exception as e:
        print(f"[ERROR] Failed during retraining pipeline: {e}")

def main():
    """
    Sentinel 스크립트의 메인 실행 로직
    """
    print(f"\n--- Running Opportunity Sentinel at {datetime.now().strftime('%Y-%m-%d %H:%M')} ---")
    universe = get_top_10_coins()
    if not universe:
        return

    for ticker in universe:
        print(f"\n[INFO] Analyzing {ticker}...")
        df_15min = pyupbit.get_ohlcv(ticker, "minute15", 96)
        df_1h = pyupbit.get_ohlcv(ticker, "minute60", 300)
        
        if df_15min is None or df_1h is None:
            print(f"[WARN] Could not fetch data for {ticker}. Skipping.")
            continue

        opportunity_segment, ts = find_missed_v_recovery(df_15min, df_1h)
        
        if opportunity_segment is not None:
            send_notification(ticker, ts)
            trigger_retraining_pipeline(ticker, opportunity_segment)
            # 한 번에 하나의 기회만 처리하고 종료
            break 
        else:
            print(f"[OK] No missed opportunities detected for {ticker}.")

if __name__ == '__main__':
    main()
