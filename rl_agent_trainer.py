import os
import shutil
import pandas as pd
from stable_baselines3 import PPO
from preprocessor import DataPreprocessor
from rl_environment import PortfolioTradingEnv
from dl_model_trainer import DLModelTrainer  # For TARGET_COINS


class RLAgentTrainer:
    """
    포트폴리오 관리를 위한 강화학습(RL) 에이전트를 훈련하고 관리합니다.
    - `MultiInputPolicy`를 사용하는 PPO 알고리즘으로 에이전트를 훈련합니다.
    - 훈련된 에이전트를 `portfolio_agent.zip` 파일로 저장하고 불러옵니다.
    """

    def __init__(self, model_path="portfolio_agent.zip"):
        self.model_path = model_path
        self.target_coins = DLModelTrainer.TARGET_COINS
        self.preprocessor = DataPreprocessor(target_coins=self.target_coins)

    def _load_all_data(self) -> dict | None:
        """훈련을 위해 모든 타겟 코인의 전처리된 데이터를 불러옵니다."""
        print("모든 타겟 코인의 데이터 로딩 중...")
        
        # Ensure preprocessed_data.pkl is up-to-date
        # This will call the run method of DataPreprocessor, which handles caching
        self.preprocessor.run() 

        # Load the combined preprocessed data
        data_path = os.path.join(self.preprocessor.cache_dir, "preprocessed_data.pkl")
        if not os.path.exists(data_path):
            print(f"오류: 전처리된 데이터 파일 '{data_path}'을 찾을 수 없습니다.")
            return None
        
        all_data = pd.read_pickle(data_path)

        if not all_data or len(all_data) < len(self.target_coins):
            print("훈련에 사용할 데이터가 충분하지 않습니다. 프로세스를 중단합니다.")
            return None

        return all_data

    def train_agent(self, total_timesteps=2_000_000):
        """
        포트폴리오 관리 RL 에이전트를 훈련합니다.
        """
        log_dir = "./portfolio_rl_tensorboard_logs/"
        if os.path.exists(log_dir):
            print(f"기존 로그 디렉토리 {log_dir}를 삭제합니다.")
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        print("포트폴리오 RL 에이전트 훈련을 시작합니다...")

        # 1. 모든 코인에 대한 데이터 로드
        data_dict = self._load_all_data()
        if not data_dict:
            return

        # 2. 포트폴리오 거래 환경 생성
        env = PortfolioTradingEnv(data_dict)

        # 3. PPO 모델 인스턴스화 (MultiInputPolicy 사용)
        # MultiInputPolicy는 Dict Observation Space를 처리하기 위해 설계되었습니다.
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
        )

        # 4. 모델 훈련
        print(f"총 {total_timesteps} 타임스텝 동안 포트폴리오 에이전트를 훈련합니다...")
        model.learn(total_timesteps=total_timesteps)

        # 5. 훈련된 모델 저장
        model.save(self.model_path)
        print(f"훈련된 포트폴리오 에이전트를 '{self.model_path}' 파일로 저장했습니다.")

    def load_agent(self, env=None):
        """
        저장된 포트폴리오 에이전트를 불러옵니다.
        """
        if os.path.exists(self.model_path):
            print(f"저장된 포트폴리오 에이전트('{self.model_path}')를 불러옵니다.")
            return PPO.load(self.model_path, env=env)
        else:
            print(
                f"저장된 에이전트 파일('{self.model_path}')을 찾을 수 없습니다. 먼저 에이전트를 훈련시켜야 합니다."
            )
            return None


if __name__ == "__main__":
    # 이 파일을 직접 실행하여 훈련을 시작할 수 있습니다.
    # 예: python rl_agent_trainer.py
    trainer = RLAgentTrainer()
    trainer.train_agent(total_timesteps=200_000)  # 테스트를 위해 타임스텝을 줄여서 실행
