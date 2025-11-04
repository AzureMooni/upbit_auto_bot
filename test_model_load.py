import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from trading_env_simple import SimpleTradingEnv # Assuming this is needed for env= argument

# Dummy data for environment creation
dummy_data = pd.DataFrame(np.random.rand(100, 10), columns=[f'col_{i}' for i in range(10)])
dummy_env = SimpleTradingEnv(dummy_data)

model_path = "bullish_market_agent.zip"

try:
    print(f"Attempting to load model from {model_path}...")
    model = PPO.load(model_path, env=dummy_env)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
