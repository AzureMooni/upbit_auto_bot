import pandas as pd
import asyncio
from strategies.grid_trading import GridTrader
from scanner import classify_market_live
from core.exchange import UpbitService

def generate_sideways_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger Bands (%B)와 RSI를 사용하여 횡보장 매매 신호를 생성합니다.
    """
    # 1. 지표 계산
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.rsi(length=14, append=True)

    # 2. 신호 생성
    # 매수: %B가 0.2보다 작고, RSI가 40보다 작을 때
    buy_conditions = (df['BBP_20_2.0'] < 0.2) & (df['RSI_14'] < 40)
    # 매도: %B가 0.8보다 크고, RSI가 60보다 클 때
    sell_conditions = (df['BBP_20_2.0'] > 0.8) & (df['RSI_14'] > 60)

    # 3. 시그널 컬럼 추가
    df['signal'] = 0.0
    df.loc[buy_conditions, 'signal'] = 1.0
    df.loc[sell_conditions, 'signal'] = -1.0

    # 4. 중복 신호 제거 (포지션 유지)
    # 매수 후 다음 매도 신호가 나올 때까지 매수 신호는 무시
    # 매도 후 다음 매수 신호가 나올 때까지 매도 신호는 무시
    position = 0
    signals = []
    for i in range(len(df)):
        if position == 0 and df['signal'].iloc[i] == 1:
            position = 1
            signals.append(1)
        elif position == 1 and df['signal'].iloc[i] == -1:
            position = -1
            signals.append(-1)
        elif position == -1 and df['signal'].iloc[i] == 1:
            position = 1
            signals.append(1)
        else:
            signals.append(0)
    df['signal'] = signals
    return df

async def run_range_grid_strategy():
    """
    실시간으로 시장 상황을 스캔하고, 'SIDEWAYS' 상태일 때
    Range Grid Trading 전략을 실행하는 메인 비동기 함수.
    """
    print("🚀 Starting Range Grid Strategy...")
    upbit_service = UpbitService()
    grid_trader = GridTrader(upbit_service, symbol="BTC/KRW", num_grids=10, total_investment=50000)

    while True:
        try:
            # 실시간 시장 상황 분류
            market_state = await classify_market_live(upbit_service)
            print(f"[{pd.Timestamp.now()}] Current market state: {market_state}")

            if market_state == "SIDEWAYS":
                if not grid_trader.is_running:
                    print("✅ Market is SIDEWAYS. Starting Grid Trader...")
                    asyncio.create_task(grid_trader.run())
                else:
                    print("✅ Market is SIDEWAYS. Grid Trader is already running.")
            else: # TRENDING or DOWNTREND
                if grid_trader.is_running:
                    print(f"❌ Market is {market_state}. Stopping Grid Trader...")
                    await grid_trader.stop()
                else:
                    print(f"❌ Market is {market_state}. Grid Trader remains stopped.")

            # 30분마다 시장 상황 재확인
            print("🕒 Waiting for 30 minutes before next market scan...")
            await asyncio.sleep(1800)

        except Exception as e:
            print(f"🔥 An error occurred in the main loop: {e}")
            # 에러 발생 시 잠시 대기 후 재시도
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(run_range_grid_strategy())
    except KeyboardInterrupt:
        print("\n⏹️ Strategy stopped by user.")