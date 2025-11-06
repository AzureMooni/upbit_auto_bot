import pandas as pd
import os
import shutil
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation

from preprocessor import DataPreprocessor
from trading_env_simple import SimpleTradingEnv
from market_regime_detector import get_market_regime_dataframe

# --- Constants ---
LOOKBACK_WINDOW = 50
DATA_PATH = "cache/preprocessed_data.pkl"
LOG_DIR_BASE = "specialist_rl_tensorboard_logs/"
MODEL_SAVE_PATH_BASE = "specialist_agent_"  # Prefix for specialist models
STATS_SAVE_PATH = "specialist_stats.json"

def train_specialist_agents(total_timesteps=100000):
    # Clean up existing log directories
    if os.path.exists(LOG_DIR_BASE):
        print(f"기존 로그 디렉토리 {LOG_DIR_BASE}를 삭제합니다.")
        shutil.rmtree(LOG_DIR_BASE)
    os.makedirs(LOG_DIR_BASE, exist_ok=True)

    # --- 1. Run Preprocessing ---
    print('데이터 로딩 및 전처리 시작...')
    preprocessor = DataPreprocessor()
    all_data_dict = preprocessor.run_and_save_to_pickle(DATA_PATH)
    print(f'전처리된 데이터 {DATA_PATH}에 저장 완료.')

    if not all_data_dict:
        print("오류: 전처리된 데이터가 없습니다. 훈련을 중단합니다.")
        return

    df = pd.concat(all_data_dict.values(), ignore_index=False)
    df.sort_index(inplace=True)

    # --- 2. Identify Market Regimes ---
    print("시장 국면을 식별합니다...")
    df_with_regimes = get_market_regime_dataframe(df)
    regimes = ['Bullish', 'Bearish', 'Sideways']
    
    specialist_stats = {}

    for regime in regimes:
        print(f"\n--- {regime} 시장 국면 전문가 에이전트 훈련 시작 ---")
        regime_df = df_with_regimes[df_with_regimes['market_regime'] == regime].copy()

        if regime_df.empty or len(regime_df) < LOOKBACK_WINDOW + 200: # Ensure enough data for indicators + lookback
            print(f"경고: {regime} 시장 국면에 충분한 데이터가 없습니다. 훈련을 건너뜁니다.")
            continue

        # Create a unique log directory for each specialist
        log_dir = os.path.join(LOG_DIR_BASE, regime.lower())
        os.makedirs(log_dir, exist_ok=True)

        print(f"{regime} 시장 국면 거래 환경을 설정합니다...")
        regime_df = regime_df.drop(columns=['market_regime'])
        env = SimpleTradingEnv(regime_df, lookback_window=LOOKBACK_WINDOW)
        env = FlattenObservation(env)
        vec_env = DummyVecEnv([lambda: env])

        print(f"{regime} 시장 국면 PPO 모델을 설정하고 훈련을 시작합니다...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            device='cpu'
        )

        print(f"모델 훈련을 시작합니다... (Total Timesteps: {total_timesteps})")
        model.learn(total_timesteps=total_timesteps)

        model_save_path = f"{MODEL_SAVE_PATH_BASE}{regime.lower()}.zip"
        print(f"훈련이 완료되었습니다. 모델을 다음 경로에 저장합니다: {model_save_path}")
        model.save(model_save_path)
        
        # Initialize stats for the regime
        specialist_stats[regime] = {'wins': 0, 'losses': 0, 'total_profit': 0.0, 'total_loss': 0.0, 'trades': 0}

    # --- 3. Fallback for Missing Models ---
    print("\n--- 훈련 후 모델 파일 검증 및 폴백 처리 ---")
    regimes = ['Bullish', 'Bearish', 'Sideways']
    fallback_model_path = f"{MODEL_SAVE_PATH_BASE}sideways.zip"
    
    # Check if the fallback model itself exists
    if not os.path.exists(fallback_model_path):
        print(f"[FATAL] 치명적 오류: 기본 폴백 모델인 {fallback_model_path}가 생성되지 않았습니다.")
        print("모든 시장 국면에 대한 데이터가 부족하여 어떤 모델도 훈련되지 않았을 가능성이 높습니다.")
        # If even the fallback is missing, we cannot proceed. Exit to fail the build.
        exit(1)

    for regime in regimes:
        model_path = f"{MODEL_SAVE_PATH_BASE}{regime.lower()}.zip"
        if not os.path.exists(model_path):
            print(f"[WARN] 경고: {regime} 모델이 생성되지 않았습니다. {fallback_model_path}을(를) 복사하여 대체합니다.")
            shutil.copy(fallback_model_path, model_path)
            # Since the model is a fallback, create a default stat entry
            if regime not in specialist_stats:
                 specialist_stats[regime] = {'wins': 0, 'losses': 0, 'total_profit': 0.0, 'total_loss': 0.0, 'trades': 0}

    # Save initial specialist stats
    with open(STATS_SAVE_PATH, 'w') as f:
        json.dump(specialist_stats, f, indent=4)
    print(f"전문가 성과 파일 {STATS_SAVE_PATH} 생성 완료.")

if __name__ == "__main__":
    train_specialist_agents(total_timesteps=150000)