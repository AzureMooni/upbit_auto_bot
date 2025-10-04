import asyncio
import time
import pandas as pd
from core.exchange import UpbitService

class BreakoutTrader:
    def __init__(self, upbit_service: UpbitService, ticker: str, allocated_capital: float):
        self.upbit_service = upbit_service
        self.ticker = ticker
        self.allocated_capital = allocated_capital
        self.position_held = False
        self.entry_price = 0.0
        self.base_currency = ticker.split('/')[0]
        self.pp = 0.0 # Pivot Point
        self.r2 = 0.0 # Second Resistance for Take Profit
        self.breakout_value = 0.0

        print(f"BreakoutTrader initialized for {self.ticker}. Allocated capital: {self.allocated_capital:,.0f} KRW.")

    def _calculate_breakout_levels(self, df: pd.DataFrame, k=0.5):
        """전일 데이터를 기반으로 피봇 포인트, 저항선, 변동성 돌파 값을 계산합니다."""
        prev_day = df.iloc[-2] # 전일 데이터
        high = prev_day['high']
        low = prev_day['low']
        close = prev_day['close']

        pp = (high + low + close) / 3
        r2 = pp + (high - low)
        breakout_val = (high - low) * k
        
        self.pp = pp
        self.r2 = r2
        self.breakout_value = breakout_val

    async def run(self, interval_seconds: int = 60): # Check every minute
        """
        변동성 돌파 전략을 실행합니다.
        """
        print(f"Starting BreakoutTrader for {self.ticker}...")
        while True:
            try:
                # Fetch daily OHLCV to calculate pivot points and breakout values
                ohlcv_daily_raw = await self.upbit_service.exchange.fetch_ohlcv(self.ticker, '1d', limit=2)
                if not ohlcv_daily_raw or len(ohlcv_daily_raw) < 2:
                    print(f"Not enough daily OHLCV data for {self.ticker}. Retrying...")
                    await asyncio.sleep(interval_seconds)
                    continue
                
                df_daily = pd.DataFrame(ohlcv_daily_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                self._calculate_breakout_levels(df_daily)
                
                current_price = await self.upbit_service.get_current_price(self.ticker)
                if current_price is None:
                    await asyncio.sleep(interval_seconds)
                    continue

                # print(f"Current price for {self.ticker}: {current_price:,.2f} KRW, PP: {self.pp:,.2f}, R2: {self.r2:,.2f}, Breakout Value: {self.breakout_value:,.2f}")

                if not self.position_held:
                    # 진입 신호: 현재 가격이 (피봇 포인트 + 변동성 돌파 값) 위에 있을 때 매수
                    if current_price > (self.pp + self.breakout_value):
                        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 🚀 Breakout UP detected for {self.ticker}. Attempting to BUY.")
                        order = await self.upbit_service.create_market_buy_order(self.ticker, self.allocated_capital)
                        if order and order.get('status') == 'closed':
                            self.position_held = True
                            self.entry_price = order.get('average', current_price)
                            print(f"🟢 BUY executed for {self.ticker} at {self.entry_price:,.2f} KRW. TP: {self.r2:,.2f}, SL: {self.pp:,.2f}.")
                        else:
                            print(f"❌ BUY order failed or not closed for {self.ticker}.")
                else: # 포지션 보유 중: 익절 또는 손절 확인
                    # 익절 조건: 현재 가격이 R2에 도달
                    if current_price >= self.r2:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 🎉 Take Profit hit for {self.ticker} at {current_price:,.2f} KRW (R2: {self.r2:,.2f}).")
                        balances = await self.upbit_service.get_all_balances()
                        amount_to_sell = balances.get(self.base_currency, {}).get('balance', 0)
                        if amount_to_sell > 0:
                            order = await self.upbit_service.create_market_sell_order(self.ticker, amount_to_sell)
                            if order and order.get('status') == 'closed':
                                print(f"🔴 SELL executed for {self.ticker} (Take Profit).")
                                self.position_held = False
                                return # 전략 종료
                        else:
                            print(f"Warning: No {self.base_currency} to sell for Take Profit.")
                            self.position_held = False # 포지션 상태 강제 초기화
                            return # 전략 종료

                    # 손절 조건: 현재 가격이 피봇 포인트 아래로 내려옴
                    elif current_price <= self.pp:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 🚨 Stop Loss hit for {self.ticker} at {current_price:,.2f} KRW (PP: {self.pp:,.2f}).")
                        balances = await self.upbit_service.get_all_balances()
                        amount_to_sell = balances.get(self.base_currency, {}).get('balance', 0)
                        if amount_to_sell > 0:
                            order = await self.upbit_service.create_market_sell_order(self.ticker, amount_to_sell)
                            if order and order.get('status') == 'closed':
                                print(f"🔴 SELL executed for {self.ticker} (Stop Loss).")
                                self.position_held = False
                                return # 전략 종료
                        else:
                            print(f"Warning: No {self.base_currency} to sell for Stop Loss.")
                            self.position_held = False # 포지션 상태 강제 초기화
                            return # 전략 종료

            except Exception as e:
                print(f"An error occurred in BreakoutTrader run loop for {self.ticker}: {e}")
            
            await asyncio.sleep(interval_seconds)

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    import asyncio

    env_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("""UPBIT_ACCESS_KEY=YOUR_ACCESS_KEY
UPBIT_SECRET_KEY=YOUR_SECRET_KEY""")
        print(f"Created a dummy .env file at {env_path}. Please replace YOUR_ACCESS_KEY and UPBIT_SECRET_KEY with actual values.")
    load_dotenv(env_path)

    async def test_breakout_trader():
        # BreakoutTrader 테스트 예시
        try:
            upbit_service = UpbitService()
            await upbit_service.connect()

            ticker = 'BTC/KRW' # 예시 티커
            order_amount_krw = 100000 # 10만원

            BreakoutTrader(upbit_service, ticker, order_amount_krw)
            # await breakout_trader.run(interval_seconds=60) # 실제 실행 시 주석 해제
            print("BreakoutTrader example setup complete. To run, integrate into main.py and ensure API keys are set.")

        except Exception as e:
            print(f"An unexpected error occurred during BreakoutTrader setup: {e}")

    asyncio.run(test_breakout_trader())