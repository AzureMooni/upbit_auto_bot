from typing import Union
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
        raise NotImplementedError("Deep Learning functionality is temporarily disabled.")

    def _build_lstm_model(self, input_shape):
        raise NotImplementedError("Deep Learning functionality is temporarily disabled.")

    def train_model(self, historical_data: dict):
        raise NotImplementedError("Deep Learning functionality is temporarily disabled.")

    def load_model(self):
        print("LSTM model loading is temporarily disabled.")
        return False

    def predict_proba(self, df_ohlcv: pd.DataFrame) -> Union[np.ndarray, None]:
        print("LSTM model prediction is temporarily disabled.")
        return None

    def predict(self, df_ohlcv: pd.DataFrame):
        """
        '매수'(레이블 1) 클래스에 대한 예측 확률을 반환합니다.
        내부적으로 predict_proba를 호출합니다.
        """
        print("LSTM model prediction is temporarily disabled.")
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
