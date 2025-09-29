import pandas as pd
import pandas_ta as ta
import os
from datetime import datetime, timedelta
import numpy as np
from dl_model_trainer import DLModelTrainer # For TARGET_COINS in main

class DataPreprocessor:
    """
    데이터 전처리를 담당하는 클래스.
    - 원본 CSV 데이터를 읽어 기술적 지표를 추가합니다.
    - 결과를 빠른 로딩을 위해 Feather 포맷으로 캐시 파일에 저장합니다.
    """
    def __init__(self, target_coins: list):
        """
        Preprocessor 인스턴스를 초기화합니다.

        Args:
            target_coins (list): 전처리할 코인 티커의 리스트 (예: ["BTC/KRW", "ETH/KRW"])
        """
        self.target_coins = target_coins
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """주어진 데이터프레임에 기술적 지표를 생성하여 추가합니다."""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close', 'volume' columns.")

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        # pandas-ta를 사용하여 다양한 기술적 지표 계산
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df['BBP_20_2.0'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=60, append=True)

        # MFI (Money Flow Index) 계산
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

    def run(self, start_date_str: str, end_date_str: str):
        """
        지정된 기간 동안의 모든 타겟 코인에 대한 전처리를 실행합니다.
        결과는 Feather 파일로 캐시 디렉토리에 저장됩니다.
        """
        print("🚀 데이터 전처리를 시작합니다. 모든 기술적 지표를 계산하고 Feather 파일로 저장합니다.")
        
        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')

        for ticker in self.target_coins:
            filename_csv = ticker.replace('/', '_') + '_1h.csv'
            filepath_csv = os.path.join(self.data_dir, filename_csv)
            filename_feather = ticker.replace('/', '_') + '_1h.feather'
            filepath_feather = os.path.join(self.cache_dir, filename_feather)

            if not os.path.exists(filepath_csv):
                print(f"- {ticker} 원본 데이터 파일 ({filepath_csv})을 찾을 수 없습니다. 건너뜁니다.")
                continue

            try:
                df = pd.read_csv(filepath_csv, index_col='timestamp', parse_dates=True)
                df = df[(df.index >= start_dt) & (df.index <= end_dt + timedelta(days=1, microseconds=-1))]

                if df.empty:
                    print(f"- {ticker} 데이터가 지정된 기간 내에 없습니다. 건너뜁니다.")
                    continue

                print(f"- {ticker} 데이터 로딩 완료 ({len(df)} 행). 지표 계산 중...")
                processed_df = self._generate_features(df.copy())
                processed_df.dropna(inplace=True)

                if not processed_df.empty:
                    processed_df.reset_index(inplace=True)
                    processed_df.to_feather(filepath_feather)
                    print(f"  ✅ {ticker} 전처리 완료 및 {len(processed_df)} 행의 데이터를 {filepath_feather}에 저장했습니다.")
                else:
                    print(f"  ❌ {ticker} 지표 계산 후 유효한 데이터가 없어 저장하지 않습니다.")

            except Exception as e:
                print(f"- {ticker} 데이터 전처리 실패: {e}")

    def load_and_preprocess_single_coin(self, ticker: str, timeframe: str) -> pd.DataFrame | None:
        """
        특정 코인의 전처리된 데이터를 로드합니다.
        캐시 파일(.feather)이 있으면 로드하고, 없으면 원본(.csv)에서 생성합니다.
        """
        filename_feather = f"{ticker.replace('/', '_')}_{timeframe}.feather"
        filepath_feather = os.path.join(self.cache_dir, filename_feather)

        if os.path.exists(filepath_feather):
            try:
                df = pd.read_feather(filepath_feather)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            except Exception as e:
                print(f"Feather 파일 로딩 실패: {e}")

        print(f"캐시 파일({filepath_feather})을 찾을 수 없거나 로딩에 실패하여, 원본 CSV 파일에서 생성을 시도합니다.")
        filename_csv = f"{ticker.replace('/', '_')}_{timeframe}.csv"
        filepath_csv = os.path.join(self.data_dir, filename_csv)

        if not os.path.exists(filepath_csv):
            print(f"원본 CSV 파일({filepath_csv})을 찾을 수 없습니다.")
            return None

        try:
            df = pd.read_csv(filepath_csv, index_col='timestamp', parse_dates=True)
            if df.empty:
                return None
            
            print(f"{ticker} 원본 데이터 로딩 완료. 지표 계산 중...")
            processed_df = self._generate_features(df.copy())
            processed_df.dropna(inplace=True)

            if not processed_df.empty:
                processed_df.reset_index(inplace=True)
                processed_df.to_feather(filepath_feather)
                print(f"  ✅ {ticker} 전처리 완료 및 캐시 파일 저장: {filepath_feather}")
                processed_df.set_index('timestamp', inplace=True)
                return processed_df
            else:
                return None
        except Exception as e:
            print(f"{ticker} CSV 파일 처리 중 에러 발생: {e}")
            return None

if __name__ == '__main__':
    # 이 파일을 직접 실행하면 모든 타겟 코인에 대한 전처리를 수행합니다.
    preprocessor = DataPreprocessor(target_coins=DLModelTrainer.TARGET_COINS)
    start_date = '2018-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    preprocessor.run(start_date, end_date)
