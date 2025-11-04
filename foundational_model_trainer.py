import pandas as pd
import os
import shutil
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation

# FIX: Correct imports for a clean environment
from preprocessor import DataPreprocessor
from trading_env_simple import SimpleTradingEnv

# --- Constants ---
LOOKBACK_WINDOW = 50
DATA_PATH = "cache/preprocessed_data.pkl"
LOG_DIR = "foundational_rl_tensorboard_logs/"
MODEL_SAVE_PATH = "foundational_agent.zip"
STATS_SAVE_PATH = "specialist_stats.json"

def train_foundational_agent(total_timesteps=100000):
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)

    print('데이터 로딩 및 전처리 시작...')
    preprocessor = DataPreprocessor()
    all_data_dict = preprocessor.run_and_save_to_pickle(DATA_PATH)

    if not all_data_dict:
        print("오류: 전처리된 데이터가 없습니다. 훈련을 중단합니다.")
        return

    df = pd.concat(all_data_dict.values(), ignore_index=False)
    df.sort_index(inplace=True) 

    print("거래 환경을 설정합니다...")
    env = SimpleTradingEnv(df, lookback_window=LOOKBACK_WINDOW)
    env = FlattenObservation(env)
    vec_env = DummyVecEnv([lambda: env])

    print("PPO 모델을 설정하고 훈련을 시작합니다...")
    model = PPO(
        "MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR, n_steps=2048, batch_size=64, n_epochs=10
    )

    print(f"모델 훈련을 시작합니다... (Total Timesteps: {total_timesteps})")
    model.learn(total_timesteps=total_timesteps)

    print(f"훈련이 완료되었습니다. 모델을 다음 경로에 저장합니다: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

    stats_file = 'specialist_stats.json'
    if not os.path.exists(stats_file):
        stats = {
            regime: {'wins': 0, 'losses': 0, 'total_profit': 0.0, 'total_loss': 0.0, 'trades': 0}
            for regime in ['Bullish', 'Bearish', 'Sideways']
        }
        with open(STATS_SAVE_PATH, 'w') as f:
            json.dump(stats, f)

if __name__ == "__main__":
    train_foundational_agent(total_timesteps=150000)
