
import pandas as pd
import lightgbm as lgb
import joblib
import os
from market_regime_detector import precompute_all_indicators

def train_price_prediction_model(data: pd.DataFrame, model_save_path: str, future_steps=12, profit_threshold=0.02):
    print("[INFO] Starting Advanced Model training process...")
    df = precompute_all_indicators(data.copy())

    df['future_price'] = df['close'].shift(-future_steps)
    df['label'] = (df['future_price'] > df['close'] * (1 + profit_threshold)).astype(int)
    df.dropna(inplace=True)

    # [FIX] Correct and complete feature column names
    feature_cols = [
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBP_20_2.0', 'BBB_20_2.0', 'ATRr_14',
        'STOCHk_14_14_3_3', 'STOCHd_14_14_3_3',
        'PPO_12_26_9', 'PPOh_12_26_9', 'PPOs_12_26_9'
    ]
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column {col} not found in DataFrame after indicator calculation.")

    X = df[feature_cols]
    y = df['label']

    print(f"[INFO] Training with {len(X)} samples and {len(feature_cols)} features.")
    lgb_clf = lgb.LGBMClassifier(objective='binary', n_estimators=1000, random_state=42)
    lgb_clf.fit(X, y)

    joblib.dump(lgb_clf, model_save_path)
    print(f"[SUCCESS] Advanced model saved to {model_save_path}")

def predict_win_probability(live_features: pd.DataFrame, model_path: str) -> float:
    if not os.path.exists(model_path): return 0.0
    model = joblib.load(model_path)
    # Ensure columns are in the same order as training
    feature_cols = [
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBP_20_2.0', 'BBB_20_2.0', 'ATRr_14',
        'STOCHk_14_14_3_3', 'STOCHd_14_14_3_3',
        'PPO_12_26_9', 'PPOh_12_26_9', 'PPOs_12_26_9'
    ]
    return float(model.predict_proba(live_features[feature_cols])[:, 1][0])
