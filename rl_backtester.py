import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation

from rl_environment import TradingEnv

# --- Constants ---
DATA_PATH = "cache/preprocessed_data.pkl"
MODEL_PATH = "foundational_agent.zip"
SYMBOL = "KRW-BTC"

def run_rl_backtest():
    """
    훈련된 RL 에이전트를 로드하여 백테스트를 실행하고, 상세 거래 내역을 포함한 성과를 평가합니다.
    """
    print(f"백테스트를 위해 훈련된 모델을 로드합니다: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 모델 파일이 없습니다 - {MODEL_PATH}")
        return
    
    model = PPO.load(MODEL_PATH)

    print(f"백테스트용 데이터를 로드합니다: {DATA_PATH}")
    df = pd.read_pickle(DATA_PATH)
    
    print("거래 환경을 설정합니다...")
    env = TradingEnv(df, symbol=SYMBOL)
    env = FlattenObservation(env)

    # --- 시뮬레이션 루프 ---
    print("백테스트 시뮬레이션을 시작합니다...")
    obs, info = env.reset()
    done = False
    portfolio_history = []
    trade_history = [] # [NEW] 거래 내역 기록
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        portfolio_history.append(info['portfolio_value'])
        # [NEW] 거래 정보가 있으면 기록
        if "trade" in info:
            trade_history.append(info["trade"])

    print("시뮬레이션이 완료되었습니다. 성과를 계산합니다...")

    # --- 상세 거래 내역 출력 ---
    print("\n--- TRADE HISTORY ---")
    if not trade_history:
        print("  No trades were executed.")
    else:
        trade_df = pd.DataFrame(trade_history)
        coin_symbol = SYMBOL.split('-')[1]
        for i, trade in trade_df.iterrows():
            print(
                f"  - [{pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M')}] "
                f"{trade['action']:<4} | "
                f"Price: {trade['price']:>11,.0f} KRW | "
                f"Amount: {trade['amount']:<10.6f} {coin_symbol}"
            )
    print("-" * 50)

    # --- 최종 성과 보고 ---
    report_df = pd.DataFrame({'portfolio_value': portfolio_history})
    initial_capital = env.unwrapped.initial_capital

    # AI 에이전트 성과
    final_portfolio_value = report_df["portfolio_value"].iloc[-1]
    total_return = (final_portfolio_value / initial_capital - 1) * 100
    rolling_max = report_df["portfolio_value"].cummax()
    daily_drawdown = report_df["portfolio_value"] / rolling_max - 1.0
    mdd = daily_drawdown.cummin().iloc[-1] * 100
    daily_returns = report_df["portfolio_value"].pct_change()
    
    # 거래 통계 계산
    total_trades = len(trade_history)
    num_days = (df.index[-1] - df.index[0]).days
    avg_trades_per_day = total_trades / num_days if num_days > 0 else 0
    
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365*24) if daily_returns.std() > 0 else 0 # 시간봉 기준 연율화

    # 벤치마크 (Buy & Hold) 성과
    benchmark_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    final_benchmark_value = initial_capital * (1 + benchmark_return / 100)

    print("\n--- 📊 RL 에이전트 최종 성과 보고 ---")
    print(f"  - 시뮬레이션 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print("-" * 50)
    print("  [RL Agent 성과]")
    print(f"  - 최종 자산: {final_portfolio_value:,.0f} KRW")
    print(f"  - 총 수익률: {total_return:.2f}%")
    print(f"  - 최대 낙폭 (MDD): {mdd:.2f}%")
    print(f"  - 샤프 지수 (연율화): {sharpe_ratio:.2f}")
    print(f"  - 총 거래 횟수: {total_trades} 회")
    print(f"  - 일일 평균 거래 횟수: {avg_trades_per_day:.2f} 회")
    print("-" * 50)
    print("  [벤치마크 (Buy & Hold) 성과]")
    print(f"  - 최종 자산: {final_benchmark_value:,.0f} KRW")
    print(f"  - 총 수익률: {benchmark_return:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    run_rl_backtest()