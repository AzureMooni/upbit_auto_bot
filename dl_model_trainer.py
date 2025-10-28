import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model_trainer import ModelTrainer  # For TARGET_COINS
from data_pipeline import DataPipeline


class DLModelTrainer:
    TARGET_COINS = [
        "BTC/KRW",
        "ETH/KRW",
        "XRP/KRW",
        "SOL/KRW",
        "DOGE/KRW",
        "AVAX/KRW",
        "LINK/KRW",
    ]

    def __init__(
        self,
        model_path="price_predictor_dl.h5",
        scaler_path="price_scaler_dl.pkl",
        sequence_length=72,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.sequence_length = sequence_length  # LSTM 입력 시퀀스 길이
        self.pipeline = DataPipeline(scaler_path=self.scaler_path)

    def load_historical_data(self, start_date_str: str, end_date_str: str):
        print("과거 데이터 로딩 중 (Feather 파일에서)... ")
        all_data = {}
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")

        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

        for ticker in self.TARGET_COINS:  # DLModelTrainer의 TARGET_COINS 사용
            filename_feather = ticker.replace("/", "_") + "_1h.feather"
            filepath_feather = os.path.join(cache_dir, filename_feather)

            if not os.path.exists(filepath_feather):
                print(
                    f"- {ticker} 캐시 파일 ({filepath_feather})을 찾을 수 없습니다. 건너뜜."
                )
                continue

            try:
                df = pd.read_feather(filepath_feather)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df = df[
                    (df.index >= start_dt)
                    & (df.index <= end_dt + timedelta(days=1, microseconds=-1))
                ]

                if not df.empty:
                    all_data[ticker] = df
                    print(f"- {ticker} 데이터 로딩 완료 ({len(df)} 행)")
                else:
                    print(f"- {ticker} 데이터가 지정된 기간 내에 없습니다. 건너뜜.")
            except Exception as e:
                print(f"- {ticker} 데이터 로딩 실패: {e}")

        return all_data

    def _create_sequences(self, data, labels):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : (i + self.sequence_length)])
            y.append(labels[i + self.sequence_length])
        return np.array(X), np.array(y)

    def _build_lstm_model(self, input_shape):
        model = Sequential(
            [
                LSTM(
                    100, return_sequences=True, input_shape=input_shape
                ),  # 첫 번째 LSTM 층
                Dropout(0.3),  # 과적합 방지를 위한 Dropout
                LSTM(50, return_sequences=False),  # 두 번째 LSTM 층
                Dropout(0.3),  # 과적합 방지를 위한 Dropout
                Dense(3, activation="softmax"),  # 출력층 (3개 뉴런: 매수, 관망, 매도)
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

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

            # _generate_features는 model_trainer.py에서 가져온다.
            # 여기서는 이미 전처리된 Feather 파일을 사용하므로,
            # _generate_features는 preprocessor.py에서 계산된 지표들을 그대로 사용한다.
            # 따라서, df_ohlcv는 이미 지표가 포함된 상태여야 한다.
            # 필요한 컬럼만 선택하여 특징으로 사용
            feature_columns = [
                col
                for col in df_ohlcv.columns
                if col
                not in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "future_close",
                    "price_change",
                    "label",
                ]
            ]
            features_df = df_ohlcv[feature_columns].copy()

            # Ensure all feature columns are numeric
            features_df = features_df.select_dtypes(include=np.number)

            # 결측값(NaN) 일괄 제거
            features_df.dropna(inplace=True)

            MIN_DATA_FOR_TRAINING = (
                self.sequence_length + 1
            )  # 시퀀스 생성에 필요한 최소 데이터 수
            if len(features_df) < MIN_DATA_FOR_TRAINING:
                print(
                    f"Skipping {ticker}: Not enough data ({len(features_df)} rows) after cleaning. Minimum required: {MIN_DATA_FOR_TRAINING}"
                )
                continue

            # 정답(Label) 생성: 1시간 뒤 0.5% 이상 변동했는지 예측
            future_period = 1
            threshold = 0.005  # 0.5% 변동 임계값
            df_ohlcv["future_close"] = df_ohlcv["close"].shift(-future_period)
            df_ohlcv["price_change"] = (
                df_ohlcv["future_close"] - df_ohlcv["close"]
            ) / df_ohlcv["close"]

            df_ohlcv["label"] = 0  # Default to Hold
            df_ohlcv.loc[df_ohlcv["price_change"] >= threshold, "label"] = 1  # Buy
            df_ohlcv.loc[df_ohlcv["price_change"] <= -threshold, "label"] = 2  # Sell

            labels = (
                df_ohlcv["label"].loc[features_df.index].copy()
            )  # features_df와 인덱스 정렬

            # Ensure features and labels have the same index
            common_index = features_df.index.intersection(labels.index)
            features_df = features_df.loc[common_index]
            labels = labels.loc[common_index]

            if not features_df.empty and not labels.empty:
                all_features.append(features_df)
                all_labels.append(labels)
            else:
                print(
                    f"Skipping {ticker}: Not enough data after feature/label generation. (Features empty: {features_df.empty}, Labels empty: {labels.empty})"
                )

        if not all_features:
            print("No sufficient data to train the model.")
            return

        X_combined = pd.concat(all_features)
        y_combined = pd.concat(all_labels)

        # 스케일링
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self.scaler.fit_transform(X_combined)
        y_scaled = y_combined.values  # LSTM은 1D 배열을 기대

        # 시퀀스 생성
        X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled)

        if len(X_sequences) == 0:
            print("No sufficient sequences to train the model.")
            return

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences,
            y_sequences,
            test_size=0.2,
            random_state=42,
            stratify=y_sequences,
        )

        # 모델 구축 및 학습
        self.model = self._build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
            ),
        ]

        self.model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )

        # 모델 및 스케일러 저장
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"LSTM model saved to {self.model_path}")
        print(f"Scaler saved to {self.scaler_path}")

    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("LSTM model and scaler loaded successfully.")
            return True
        print("LSTM model or scaler files not found.")
        return False

    def predict_proba(self, df_ohlcv: pd.DataFrame) -> np.ndarray | None:
        """
        데이터 파이프라인을 사용하여 데이터를 처리하고, [관망, 매수, 매도] 확률을 예측합니다.
        """
        if self.model is None:
            if not self.load_model():
                print("예측 불가: 모델이 로드되지 않았습니다.")
                return None

        # 데이터 파이프라인을 통해 예측용 데이터 시퀀스 생성
        processed_sequence = self.pipeline.process_for_prediction(
            df_ohlcv, self.sequence_length
        )

        if processed_sequence is None:
            # print("데이터 처리 실패로 예측을 진행할 수 없습니다.") # 파이프라인 내부에서 이미 로그 출력
            return None

        # 모델 예측
        probabilities = self.model.predict(processed_sequence)[0]
        return probabilities

    def predict(self, df_ohlcv: pd.DataFrame):
        """
        '매수'(레이블 1) 클래스에 대한 예측 확률을 반환합니다.
        내부적으로 predict_proba를 호출합니다.
        """
        probabilities = self.predict_proba(df_ohlcv)
        if probabilities is not None:
            return probabilities[1]  # '매수' 확률 반환
        return None


if __name__ == "__main__":
    print("--- DLModelTrainer Example ---")
    # 이 부분은 실제 사용 시 main.py에서 호출됩니다.
    # 예시를 위해 더미 데이터 로딩 및 학습/예측 과정을 보여줍니다.

    # 1. 더미 데이터 생성 (preprocessor.py에서 생성된 Feather 파일과 유사한 형태)
    dummy_data = {}
    for i in range(2):  # 2 dummy coins
        ticker = f"COIN{i}/KRW"
        data_points = 2000  # 충분한 데이터 포인트
        dates = pd.date_date_range(start="2023-01-01", periods=data_points, freq="H")
        df = pd.DataFrame(
            {
                "open": np.random.rand(data_points) * 1000 + 10000,
                "high": np.random.rand(data_points) * 1000 + 10500,
                "low": np.random.rand(data_points) * 1000 + 9500,
                "close": np.random.rand(data_points) * 1000 + 10000,
                "volume": np.random.rand(data_points) * 100000,
            },
            index=dates,
        )

        # pandas-ta 지표 추가 (preprocessor.py에서 수행되는 과정 모방)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df["BBP_20_2.0"] = (df["close"] - df["BBL_20_2.0"]) / (
            df["BBU_20_2.0"] - df["BBL_20_2.0"]
        )
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=60, append=True)

        # MFI (Money Flow Index) - Manual Calculation
        df["TP"] = (df["high"] + df["low"] + df["close"]) / 3
        df["MF"] = df["TP"] * df["volume"]
        df["PMF"] = 0.0
        df["NMF"] = 0.0
        df["prev_TP"] = df["TP"].shift(1)
        df.loc[df["TP"] > df["prev_TP"], "PMF"] = df["MF"]
        df.loc[df["TP"] < df["prev_TP"], "NMF"] = df["MF"]
        mfi_period = 14
        df["PMF_sum"] = df["PMF"].rolling(window=mfi_period).sum()
        df["NMF_sum"] = df["NMF"].rolling(window=mfi_period).sum()
        df["MFR"] = df["PMF_sum"] / df["NMF_sum"].replace(0, np.nan)
        df["MFI_14"] = 100 - (100 / (1 + df["MFR"]))
        df.drop(
            columns=["TP", "MF", "PMF", "NMF", "prev_TP", "PMF_sum", "NMF_sum", "MFR"],
            inplace=True,
            errors="ignore",
        )

        dummy_data[ticker] = df

    # 2. 모델 학습
    trainer = DLModelTrainer()
    # trainer.train_model(dummy_data) # 실제 학습 시 주석 해제

    # 3. 모델 로드 및 예측
    if os.path.exists("lstm_price_predictor.h5") and os.path.exists(
        "lstm_price_scaler.pkl"
    ):
        predictor = DLModelTrainer()
        if predictor.load_model():
            sample_df = dummy_data["COIN0/KRW"].iloc[
                -100:
            ]  # 예측에 필요한 충분한 데이터
            buy_probability = predictor.predict(sample_df)
            if buy_probability is not None:
                print(
                    f"Prediction for COIN0/KRW: Probability of Buy (label 1) in next 1 hour: {buy_probability:.4f}"
                )
            else:
                print("Prediction failed.")
    else:
        print("Model files not found. Please run training first.")
