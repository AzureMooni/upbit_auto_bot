import pandas as pd
import numpy as np
import os
import joblib
import pandas_ta as ta

class DataPipeline:
    """
    학습과 예측 전반에 걸쳐 데이터 처리 과정을 표준화하는 데이터 파이프라인.
    - 예측 요청 시, 학습에 사용된 것과 동일한 기술적 지표를 생성합니다.
    - 학습된 스케일러를 사용하여 데이터를 정규화합니다.
    - 최종적으로 모델에 입력 가능한 시퀀스 형태로 데이터를 변환합니다.
    """
    def __init__(self, scaler_path='price_scaler_dl.pkl'):
        self.scaler_path = scaler_path
        self.scaler = self._load_scaler()
        self.feature_columns = None
        if self.scaler and hasattr(self.scaler, 'feature_names_in_'):
            self.feature_columns = self.scaler.feature_names_in_

    def _load_scaler(self):
        """저장된 스케일러를 불러옵니다."""
        try:
            if os.path.exists(self.scaler_path):
                print("데이터 파이프라인: 스케일러를 성공적으로 불러왔습니다.")
                return joblib.load(self.scaler_path)
            else:
                print(f"경고: 스케일러 파일({self.scaler_path})을 찾을 수 없습니다. 데이터 정규화 없이 진행됩니다.")
                return None
        except Exception as e:
            print(f"스케일러 로딩 중 에러 발생: {e}")
            return None

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """주어진 데이터프레임에 기술적 지표를 생성하여 추가합니다. (preprocessor.py와 동일)"""
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df['BBP_20_2.0'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=60, append=True)

        df['TP'] = (df['high'] + df['low'] + df['close']) / 3
        df['MF'] = df['TP'] * df['volume']
        df['prev_TP'] = df['TP'].shift(1)
        df['PMF'] = df.apply(lambda row: row['MF'] if row['TP'] > row['prev_TP'] else 0, axis=1)
        df['NMF'] = df.apply(lambda row: row['MF'] if row['TP'] < row['prev_TP'] else 0, axis=1)
        mfi_period = 14
        pmf_sum = df['PMF'].rolling(window=mfi_period).sum()
        nmf_sum = df['NMF'].rolling(window=mfi_period).sum()
        mfr = pmf_sum / nmf_sum.replace(0, np.nan)
        df['MFI_14'] = 100 - (100 / (1 + mfr))
        df.drop(columns=['TP', 'MF', 'prev_TP', 'PMF', 'NMF'], inplace=True, errors='ignore')
        
        return df

    def process_for_prediction(self, df: pd.DataFrame, sequence_length: int = 72) -> np.ndarray | None:
        """
        예측을 위해 원본 OHLCV 데이터프레임을 가공, 정규화, 시퀀스화합니다.
        """
        if self.scaler is None:
            print("데이터 처리 불가: 스케일러가 로드되지 않았습니다.")
            return None

        # 1. 기술적 지표 생성
        features_df = self._generate_features(df.copy())
        features_df.dropna(inplace=True)

        if len(features_df) < sequence_length:
            print(f"시퀀스 생성에 데이터가 충분하지 않습니다. 필요: {sequence_length}, 보유: {len(features_df)}")
            return None

        # 2. 특징 선택 및 정규화
        if self.feature_columns is None:
            print("경고: 스케일러에 특징 이름 정보가 없습니다. 데이터에서 추론합니다.")
            self.feature_columns = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

        latest_data = features_df.iloc[-sequence_length:]
        
        if not all(col in latest_data.columns for col in self.feature_columns):
            print("에러: 예측용 데이터에 스케일러가 필요로 하는 특징 컬럼이 일부 없습니다.")
            return None
            
        latest_data_for_scaling = latest_data[self.feature_columns]
        scaled_data = self.scaler.transform(latest_data_for_scaling)

        # 3. 시퀀스 형태로 변환 (batch_size, sequence_length, num_features)
        sequence = np.array([scaled_data])
        
        return sequence
