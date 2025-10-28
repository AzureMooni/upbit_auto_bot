import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class SimpleTradingEnv(gym.Env):
    """
    단일 자산 거래를 위한 간단한 강화학습 환경입니다.
    Foundational Model 훈련에 사용됩니다.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookback_window: int = 50,
        initial_balance: float = 1_000_000,
    ):
        super().__init__()

        self.df = df.dropna().reset_index(drop=True)
        print(f"[SimpleTradingEnv] Initial df length after dropna: {len(self.df)}")
        if self.df.empty:
            raise ValueError("DataFrame is empty after dropping NaN values in SimpleTradingEnv.")
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.n_features = self.df.shape[1] # Use self.df after dropna
        self.end_step = len(self.df) - 1

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: (lookback_window, num_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, self.n_features),
            dtype=np.float32,
        )



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.current_step = self.lookback_window
        obs = self._get_observation()
        print(f"[SimpleTradingEnv] Reset observation shape: {obs.shape}")
        return obs, {}

    def step(self, action):
        self.current_step += 1
        current_price = self.df["close"].iloc[self.current_step]
        old_net_worth = self.net_worth

        self._take_action(action, current_price)

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - old_net_worth

        terminated = (
            self.net_worth <= self.initial_balance * 0.5
            or self.current_step >= self.end_step
        )
        truncated = False

        obs = self._get_observation()
        print(f"[SimpleTradingEnv] Step observation shape: {obs.shape}")
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        obs = self.df.iloc[
            self.current_step - self.lookback_window : self.current_step
        ].values.astype(np.float32)
        if np.isnan(obs).any() or np.isinf(obs).any():
            print("[SimpleTradingEnv] WARNING: NaN or Inf found in observation!")
        return obs

    def _take_action(self, action, current_price):
        if action == 1:  # Buy
            if self.balance > 10:
                self.shares_held += self.balance / current_price
                self.balance = 0
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
