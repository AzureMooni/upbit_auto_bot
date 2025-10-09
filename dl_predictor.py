import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import joblib

def train_price_prediction_model(data: pd.DataFrame, model_save_path: str, future_steps=12, profit_threshold=0.02):
    """
    LightGBM 모델을 훈련하여 미래 가격 상승 확률을 예측합니다.
    모든 코인 데이터를 사용하여 하나의 일반화된 모델을 훈련합니다.
    """
    print("[INFO] Starting LightGBM model training process...")
    
    all_features = []
    all_labels = []

    for ticker, df_ticker in data.groupby('ticker'):
        print(f"[INFO] Processing features and labels for {ticker}...")
        df = df_ticker.copy()
        
        # --- Manual Indicator Calculations ---
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = macd_line - signal_line

        # Bollinger Bands %B
        mid_band = df['close'].rolling(window=20).mean()
        std_dev = df['close'].rolling(window=20).std()
        upper_band = mid_band + (std_dev * 2)
        lower_band = mid_band - (std_dev * 2)
        df['BBP'] = (df['close'] - lower_band) / (upper_band - lower_band)

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        df.dropna(inplace=True)

        # Labels (y)
        df['future_price'] = df['close'].shift(-future_steps)
        df['label'] = (df['future_price'] > df['close'] * (1 + profit_threshold)).astype(int)
        df.dropna(inplace=True)

        feature_cols = ['RSI', 'MACD_hist', 'BBP', 'ATR']
        all_features.append(df[feature_cols])
        all_labels.append(df['label'])

    X = pd.concat(all_features)
    y = pd.concat(all_labels)

    print(f"[INFO] Total training samples: {len(X)}")

    # LightGBM 모델 훈련
    print("[INFO] Training LightGBM model...")
    lgb_clf = lgb.LGBMClassifier(objective='binary', n_estimators=1000, random_state=42)
    lgb_clf.fit(X, y, eval_set=[(X, y)], callbacks=[lgb.early_stopping(10, verbose=False)])

    joblib.dump(lgb_clf, model_save_path)
    print(f"[SUCCESS] Model training complete. Model saved to {model_save_path}")

def predict_win_probability(live_data_features: pd.DataFrame, model_path: str) -> float:
    if not os.path.exists(model_path):
        return 0.0
    try:
        model = joblib.load(model_path)
        win_prob = model.predict_proba(live_data_features)[:, 1][0]
        return float(win_prob)
    except Exception as e:
        print(f"[ERROR] Failed to predict win probability: {e}")
        return 0.0