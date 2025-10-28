import pandas as pd
import os
import shutil
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation
from preprocessor import DataPreprocessor
from trading_env_simple import SimpleTradingEnv
import argparse
from sklearn.preprocessing import MinMaxScaler
import joblib

def train_foundational_agent(
    total_timesteps=100000,
    data_path="cache/preprocessed_data.pkl",
    log_dir="foundational_rl_tensorboard_logs/",
    model_save_path="foundational_agent.zip",
    stats_save_path="specialist_stats.json",
):
    # Create absolute paths
    data_path = os.path.join(os.getcwd(), data_path)
    log_dir = os.path.join(os.getcwd(), log_dir)
    model_save_path = os.path.join(os.getcwd(), model_save_path)
    stats_save_path = os.path.join(os.getcwd(), stats_save_path)

    if os.path.exists(log_dir):
        print(f"기존 로그 디렉토리 {log_dir}를 삭제합니다.")
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    print('데이터 로딩 및 전처리 시작...')
    preprocessor = DataPreprocessor(cache_dir=os.path.dirname(data_path))
    all_data_dict = preprocessor.run_and_save_to_pickle(data_path)
    print(f'전처리된 데이터 {data_path}에 저장 완료.')

    if not all_data_dict:
        print("오류: 전처리된 데이터가 없습니다. 훈련을 중단합니다.")
        return

    df = pd.concat(all_data_dict.values(), ignore_index=False)
    df.sort_index(inplace=True)

    # Scale features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Fit scaler on all features except 'regime' if it's a categorical feature
    # Assuming 'regime' is the last column and is already mapped to numerical values (0, 1, 2)
    # If 'regime' is not the last column, adjust slicing accordingly
    feature_cols = [col for col in df.columns if col != 'regime']
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save the scaler
    scaler_save_path = f"{model_save_path}.scaler"
    joblib.dump(scaler, scaler_save_path)
    print(f"스케일러가 다음 경로에 저장되었습니다: {scaler_save_path}")

    print("거래 환경을 설정합니다...")
    env = SimpleTradingEnv(df_scaled, lookback_window=50)
    env = FlattenObservation(env)
    vec_env = DummyVecEnv([lambda: env])

    print("PPO 모델을 설정하고 훈련을 시작합니다...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=2048,
        batch_size=64,
        n_epochs=10
    )

    print(f"모델 훈련을 시작합니다... (Total Timesteps: {total_timesteps})")
    model.learn(total_timesteps=total_timesteps)

    print(f"훈련이 완료되었습니다. 모델을 다음 경로에 저장합니다: {model_save_path}")
    model.save(model_save_path)

    if not os.path.exists(stats_save_path):
        print(f'{stats_save_path}이(가) 없어 새로 생성합니다.')
        stats = {
            regime: {'wins': 0, 'losses': 0, 'total_profit': 0.0, 'total_loss': 0.0, 'trades': 0}
            for regime in ['Bullish', 'Bearish', 'Sideways']
        }
        with open(stats_save_path, 'w') as f:
            json.dump(stats, f)
    print(f"기본 성과 파일 {stats_save_path} 생성 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=150000)
    parser.add_argument('--data_path', type=str, default="cache/preprocessed_data.pkl")
    parser.add_argument('--log_dir', type=str, default="foundational_rl_tensorboard_logs/")
    parser.add_argument('--model_save_path', type=str, default="foundational_agent.zip")
    parser.add_argument('--stats_save_path', type=str, default="specialist_stats.json")
    args = parser.parse_args()

    train_foundational_agent(
        total_timesteps=args.total_timesteps,
        data_path=args.data_path,
        log_dir=args.log_dir,
        model_save_path=args.model_save_path,
        stats_save_path=args.stats_save_path,
    )