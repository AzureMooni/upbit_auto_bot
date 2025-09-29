import pandas as pd
import pandas_ta as ta
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta

class ModelTrainer:
    TARGET_COINS = ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW", "AVAX/KRW", "LINK/KRW"] # 학습 대상 코인 리스트

    def __init__(self, model_path='price_predictor.pkl', scaler_path='price_scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

    def load_historical_data(self, start_date_str: str, end_date_str: str):
        print("과거 데이터 로딩 중 (Feather 파일에서)... ")
        all_data = {}
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')

        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')

        for ticker in self.TARGET_COINS:
            filename_feather = ticker.replace('/', '_') + '_1h.feather'
            filepath_feather = os.path.join(cache_dir, filename_feather)

            if not os.path.exists(filepath_feather):
                print(f"- {ticker} 캐시 파일 ({filepath_feather})을 찾을 수 없습니다. 건너뜁니다.")
                continue

            try:
                df = pd.read_feather(filepath_feather)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df[(df.index >= start_dt) & (df.index <= end_dt + timedelta(days=1, microseconds=-1))]

                if not df.empty:
                    all_data[ticker] = df
                    print(f"- {ticker} 데이터 로딩 완료 ({len(df)} 행)")
                else:
                    print(f"- {ticker} 데이터가 지정된 기간 내에 없습니다. 건너뜁니다.")
            except Exception as e:
                print(f"- {ticker} 데이터 로딩 실패: {e}")
        
        return all_data

    def _generate_features(self, df: pd.DataFrame):
        # Ensure the DataFrame has the required columns
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close', 'volume' columns.")

        # Explicitly convert OHLCV columns to float64 for pandas-ta compatibility
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        # Calculate various technical indicators using pandas_ta
        # Over 30 indicators as requested
        # Calculate core technical indicators using pandas_ta
        # 1. RSI (14기간)
        df.ta.rsi(length=14, append=True)

        # 2. MACD (12, 26, 9 설정)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # 3. 볼린저 밴드 (20기간, 표준편차 2)
        df.ta.bbands(length=20, std=2, append=True)
        # 현재 가격이 상단/하단 밴드와 얼마나 떨어져 있는지 비율 (BBP - Bollinger Band Percentage)
        df['BBP_20_2.0'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])

        # 4. ADX (14기간)
        df.ta.adx(length=14, append=True)

        # 5. ATR (14기간)
        df.ta.atr(length=14, append=True)

        # 6. 이동평균선 (20기간, 60기간)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=60, append=True)

        # MFI (Money Flow Index) - Manual Calculation to avoid dtype issues
        # 1. Calculate Typical Price
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3

        # 2. Calculate Money Flow
        df['MF'] = df['TP'] * df['volume']

        # 3. Calculate Positive and Negative Money Flow
        df['PMF'] = 0.0
        df['NMF'] = 0.0

        # Shift TP to compare with previous TP
        df['prev_TP'] = df['TP'].shift(1)

        # Calculate PMF and NMF based on TP change
        df.loc[df['TP'] > df['prev_TP'], 'PMF'] = df['MF']
        df.loc[df['TP'] < df['prev_TP'], 'NMF'] = df['MF']

        # Sum PMF and NMF over a period (e.g., 14 periods for MFI)
        mfi_period = 14
        df['PMF_sum'] = df['PMF'].rolling(window=mfi_period).sum()
        df['NMF_sum'] = df['NMF'].rolling(window=mfi_period).sum()

        # 4. Calculate Money Flow Ratio
        # Avoid division by zero
        df['MFR'] = df['PMF_sum'] / df['NMF_sum'].replace(0, np.nan) # Replace 0 with NaN to avoid division by zero

        # 5. Calculate Money Flow Index
        df['MFI_14'] = 100 - (100 / (1 + df['MFR']))

        # Clean up temporary columns
        df.drop(columns=['TP', 'MF', 'PMF', 'NMF', 'prev_TP', 'PMF_sum', 'NMF_sum', 'MFR'], inplace=True)

        # Dynamically select indicator columns as features
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        # Get all columns after indicator calculations
        all_cols = df.columns.tolist()
        # Filter out original OHLCV columns and 'timestamp' if it somehow remained
        feature_columns = [col for col in all_cols if col not in ohlcv_cols and col != 'timestamp']
        
        features_df = df[feature_columns]
        
        # Ensure all feature columns are numeric
        features_df = features_df.select_dtypes(include=np.number)
        
        # 1. 결측값(NaN) 일괄 제거
        features_df.dropna(inplace=True)

        # 2. 최소 데이터 확보 로직 추가
        MIN_DATA_FOR_TRAINING = 1000 # 학습에 필요한 최소 데이터 수
        if len(features_df) < MIN_DATA_FOR_TRAINING:
            print(f"Warning: Not enough data ({len(features_df)} rows) after cleaning. Minimum required: {MIN_DATA_FOR_TRAINING}")
            return pd.DataFrame() # 빈 DataFrame 반환

        return features_df

    def _generate_labels(self, df: pd.DataFrame, future_period=1, threshold=0.01):
        # Calculate future price change
        df['future_close'] = df['close'].shift(-future_period)
        df['price_change'] = (df['future_close'] - df['close']) / df['close']

        # Generate labels
        # 1: Buy (price increases by threshold or more)
        # 2: Sell (price decreases by threshold or more)
        # 0: Hold (price change is within -threshold and +threshold)
        df['label'] = 0 # Default to Hold
        df.loc[df['price_change'] >= threshold, 'label'] = 1 # Buy
        df.loc[df['price_change'] <= -threshold, 'label'] = 2 # Sell

        df.dropna(inplace=True)
        return df['label']

    def train_model(self, historical_data: dict):
        all_features = []
        all_labels = []

        for ticker, df_ohlcv in historical_data.items():
            if df_ohlcv.empty:
                print(f"Skipping {ticker}: No OHLCV data.")
                continue
            
            print(f"Processing {ticker}: Initial OHLCV data length: {len(df_ohlcv)}")

            # Ensure index is datetime for pandas_ta
            if not isinstance(df_ohlcv.index, pd.DatetimeIndex):
                df_ohlcv.index = pd.to_datetime(df_ohlcv.index)

            # Generate features
            features = self._generate_features(df_ohlcv.copy()) # Use a copy to avoid SettingWithCopyWarning
            print(f"Processing {ticker}: Features data length: {len(features)}")
            
            # Align features and original DataFrame for label generation
            aligned_df = df_ohlcv.loc[features.index]
            labels = self._generate_labels(aligned_df.copy()) # Use a copy
            print(f"Processing {ticker}: Labels data length: {len(labels)}")

            # Ensure features and labels have the same index
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            print(f"Processing {ticker}: Common index length: {len(common_index)}")

            if not features.empty and not labels.empty:
                all_features.append(features)
                all_labels.append(labels)
            else:
                print(f"Skipping {ticker}: Not enough data after feature/label generation. (Features empty: {features.empty}, Labels empty: {labels.empty})")

        if not all_features:
            print("No sufficient data to train the model.")
            return

        X = pd.concat(all_features)
        y = pd.concat(all_labels)

        # Drop any remaining NaN values from X and y, ensuring alignment
        combined = pd.concat([X, y], axis=1)
        combined.dropna(inplace=True)
        X = combined.iloc[:, :-1] # All columns except the last one
        y = combined.iloc[:, -1]  # The last column is y

        if X.empty or y.empty:
            print("No sufficient data to train the model after dropping NaNs.")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train XGBoost Classifier
        self.model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model training complete. Accuracy: {accuracy:.4f}")

        # Save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model saved to {self.model_path}")
        print(f"Scaler saved to {self.scaler_path}")

    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("Model and scaler loaded successfully.")
            return True
        print("Model or scaler files not found.")
        return False

    def predict(self, df_ohlcv: pd.DataFrame):
        if self.model is None or self.scaler is None:
            if not self.load_model():
                print("Cannot make predictions: Model or scaler not loaded.")
                return None

        features = self._generate_features(df_ohlcv.copy())
        if features.empty:
            print("Not enough data to generate features for prediction.")
            return None
        
        # Ensure feature columns match those used during training
        # This is a crucial step to prevent errors during prediction
        # For simplicity, we'll assume the order and names are consistent.
        # In a real-world scenario, you might need to reindex or reorder features.
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(features_scaled)
        
        # Return the probability of 'buy' (label 1) for the latest data point
        # Assuming the last row of features corresponds to the latest data
        if probabilities.shape[0] > 0:
            return probabilities[-1, 1] # Probability of label 1 (Buy)
        return None

if __name__ == '__main__':
    # Example usage for training
    print("--- ModelTrainer Example: Training ---")
    # You would typically load real historical data here
    # For demonstration, let's create some dummy data
    dummy_data = {}
    for i in range(5): # 5 dummy coins
        ticker = f"COIN{i}/KRW"
        # Generate more data points to satisfy indicator requirements
        data_points = 500 
        dates = pd.date_range(start='2023-01-01', periods=data_points, freq='H')
        df = pd.DataFrame({
            'open': np.random.rand(data_points) * 1000 + 10000,
            'high': np.random.rand(data_points) * 1000 + 10500,
            'low': np.random.rand(data_points) * 1000 + 9500,
            'close': np.random.rand(data_points) * 1000 + 10000,
            'volume': np.random.rand(data_points) * 100000
        }, index=dates)
        # Make prices trend up or down slightly to create some labels
        df['close'] = df['close'] + np.arange(data_points) * (0.1 if i % 2 == 0 else -0.1)
        df['open'] = df['open'] + np.arange(data_points) * (0.1 if i % 2 == 0 else -0.1)
        df['high'] = df['high'] + np.arange(data_points) * (0.1 if i % 2 == 0 else -0.1)
        df['low'] = df['low'] + np.arange(data_points) * (0.1 if i % 2 == 0 else -0.1)
        dummy_data[ticker] = df

    trainer = ModelTrainer()
    # trainer.train_model(dummy_data) # Uncomment to run training

    print("""
--- ModelTrainer Example: Prediction ---""")
    if os.path.exists('price_predictor.pkl') and os.path.exists('price_scaler.pkl'):
        trainer_predictor = ModelTrainer()
        if trainer_predictor.load_model():
            # Use the last coin's data for prediction example
            sample_df = dummy_data['COIN0/KRW'].iloc[-100:] # Need enough data for indicators
            buy_probability = trainer_predictor.predict(sample_df)
            if buy_probability is not None:
                print(f"Prediction for COIN0/KRW: Probability of Buy (label 1) in next 6 hours: {buy_probability:.4f}")
            else:
                print("Prediction failed.")
    else:
        print("Model files not found. Please run training first.")