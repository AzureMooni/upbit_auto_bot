
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class TradingEnv(gym.Env):
    """
    강화학습 에이전트를 위한 커스텀 암호화폐 거래 환경.
    상세 거래 기록 로깅 기능이 추가되었습니다.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, symbol="KRW-BTC", lookback_window=50, initial_capital=1_000_000, transaction_cost=0.0005):
        super(TradingEnv, self).__init__()

        self.df = df
        self.symbol = symbol
        self.lookback_window = lookback_window
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(lookback_window, 13), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        
        self.INACTIVITY_THRESHOLD = 12
        self.INACTIVITY_PENALTY = -0.01
        self.PROFIT_BONUS_FACTOR = 0.5
        self.TARGET_SORTINO = 1.5
        self.SORTINO_BONUS = 10.0
        
    def _get_observation(self):
        obs = self.df.iloc[self.current_step - self.lookback_window + 1 : self.current_step + 1]
        return obs.values

    def _get_info(self):
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings_value": self.holdings * self.df.iloc[self.current_step]["close"],
            "entry_price": self.entry_price
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window - 1
        self.cash = self.initial_capital
        self.holdings = 0
        self.portfolio_value = self.initial_capital
        self.entry_price = 0
        self.consecutive_holds = 0
        self.episode_trade_returns = []

        return self._get_observation(), self._get_info()

    def _execute_trade(self, action):
        trade_log = None
        trade_return = 0.0

        current_price = self.df.iloc[self.current_step]["close"]
        timestamp = self.df.index[self.current_step]

        if action == 1: # 매수
            if self.cash > 0:
                amount_to_buy_krw = self.cash * (1 - self.transaction_cost)
                self.holdings = amount_to_buy_krw / current_price
                self.cash = 0
                self.entry_price = current_price
                trade_log = {
                    'timestamp': timestamp, 'action': 'BUY', 'price': current_price, 'amount': self.holdings
                }
        
        elif action == 2: # 매도
            if self.holdings > 0:
                sell_value = self.holdings * current_price * (1 - self.transaction_cost)
                if self.entry_price > 0:
                    trade_return = (current_price - self.entry_price) / self.entry_price
                
                trade_log = {
                    'timestamp': timestamp, 'action': 'SELL', 'price': current_price, 'amount': self.holdings
                }
                self.cash = sell_value
                self.holdings = 0
                self.entry_price = 0
                if trade_return != 0.0:
                    self.episode_trade_returns.append(trade_return)

        return trade_return, trade_log

    def _calculate_reward(self, portfolio_return: float, action: int, trade_return: float) -> float:
        reward = portfolio_return
        if action == 0:
            self.consecutive_holds += 1
        else:
            self.consecutive_holds = 0
        if self.consecutive_holds > self.INACTIVITY_THRESHOLD:
            reward += self.INACTIVITY_PENALTY
        if trade_return > 0:
            reward += self.PROFIT_BONUS_FACTOR * trade_return
        return reward

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        portfolio_value_before_trade = self.portfolio_value
        trade_return, trade_log = self._execute_trade(action)

        current_price = self.df.iloc[self.current_step]["close"]
        self.portfolio_value = self.cash + self.holdings * current_price
        
        portfolio_return = (self.portfolio_value / portfolio_value_before_trade) - 1 if portfolio_value_before_trade > 0 else 0.0
        reward = self._calculate_reward(portfolio_return, action, trade_return)

        if done:
            returns = self.episode_trade_returns
            if len(returns) > 1:
                downside_returns = [r for r in returns if r < 0]
                downside_deviation = np.std(downside_returns) if len(downside_returns) > 1 else 0
                mean_return = np.mean(returns)
                if downside_deviation > 1e-6:
                    sortino_ratio = mean_return / downside_deviation
                    if sortino_ratio > self.TARGET_SORTINO:
                        reward += self.SORTINO_BONUS

        observation = self._get_observation()
        info = self._get_info()
        if trade_log:
            info['trade'] = trade_log

        return observation, reward, done, False, info
