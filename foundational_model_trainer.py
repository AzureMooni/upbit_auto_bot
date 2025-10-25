import pandas as pd
import os
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation

from preprocessor import DataPreprocessor
from data_fetcher import DataFetcher
import os

from rl_environment import TradingEnv
from preprocessor import DataPreprocessor

# --- Constants ---
LOOKBACK_WINDOW = 50
DATA_PATH = "cache/preprocessed_data.pkl"
LOG_DIR = "foundational_rl_tensorboard_logs/"
MODEL_SAVE_PATH = "foundational_agent.zip"

def train_foundational_agent(total_timesteps=100000):
    """
    Trains the foundational PPO agent on the preprocessed data.
    """
    if os.path.exists(LOG_DIR):
        print(f"기존 로그 디렉토리 {LOG_DIR}를 삭제합니다.")
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("전처리된 캐시 데이터를 불러옵니다...")
    # Run preprocessing to ensure preprocessed_data.pkl is created
    fetcher = DataFetcher()
    all_raw_data = fetcher.load_all_data_from_disk_or_api() # 데이터 로드
    preprocessor = DataPreprocessor(all_raw_data)
    all_data = preprocessor.run_and_save_to_pickle(DATA_PATH) # 전처리 실행 및 .pkl 파일 저장

    all_data = pd.read_pickle(DATA_PATH)

    # Concatenate all dataframes into a single dataframe for foundational model training
    # Assuming all_data is a dictionary of {ticker: dataframe}
    if isinstance(all_data, dict) and all_data:
        df = pd.concat(all_data.values(), ignore_index=False)
        df.sort_index(inplace=True) # Ensure chronological order
    else:
        print("오류: 전처리된 데이터가 비어 있거나 예상 형식이 아닙니다.")
        return

    print("거래 환경을 설정합니다...")
    env = TradingEnv(df, lookback_window=LOOKBACK_WINDOW)
    env = FlattenObservation(env)
    vec_env = DummyVecEnv([lambda: env])

    print("PPO 모델을 설정하고 훈련을 시작합니다...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,  # This automatically handles TensorBoard logging
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    print("모델 훈련을 시작합니다... (최종 수정)")
    # The `callback` argument is removed as it's handled by `tensorboard_log`
    model.learn(total_timesteps=total_timesteps)

    print(f"훈련이 완료되었습니다. 모델을 다음 경로에 저장합니다: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_foundational_agent(total_timesteps=150000)