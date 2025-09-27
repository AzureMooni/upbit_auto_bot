import time
import pandas as pd
import ccxt
from core.exchange import UpbitService
from scanner import _calculate_breakout_levels_live # Helper for pivot points and breakout values

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

    def run(self, interval_seconds: int = 60): # Check every minute
        """
        변동성 돌파 전략을 실행합니다.
        """
        print(f"Starting BreakoutTrader for {self.ticker}...")
        while True:
            try:
                # Fetch daily OHLCV to calculate pivot points and breakout values
                ohlcv_daily = self.upbit_service.exchange.fetch_ohlcv(self.ticker, '1d', limit=2)
                if not ohlcv_daily or len(ohlcv_daily) < 2:
                    print(f"Not enough daily OHLCV data for {self.ticker}. Retrying...")
                    time.sleep(interval_seconds)
                    continue
                
                self.pp, r1, s1, self.r2, s2, self.breakout_value = _calculate_breakout_levels_live(ohlcv_daily)
                
                if self.pp is None:
                    print(f"Could not calculate pivot points for {self.ticker}. Retrying...")
                    time.sleep(interval_seconds)
                    continue

                current_price = self.upbit_service.get_current_price(self.ticker)
                if current_price is None:
                    print(f"Could not fetch current price for {self.ticker}. Retrying...")
                    time.sleep(interval_seconds)
                    continue

                print(f"Current price for {self.ticker}: {current_price:,.2f} KRW, PP: {self.pp:,.2f}, R2: {self.r2:,.2f}, Breakout Value: {self.breakout_value:,.2f}")

                if not self.position_held:
                    # 진입 신호: 현재 가격이 (피봇 포인트 + 변동성 돌파 값) 위에 있을 때 매수
                    if current_price > (self.pp + self.breakout_value):
                        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 🚀 Breakout UP detected for {self.ticker}. Attempting to BUY.")
                        order = self.upbit_service.create_market_buy_order(self.ticker, self.allocated_capital)
                        if order and order['status'] == 'closed':
                            self.position_held = True
                            self.entry_price = current_price
                            print(f"🟢 BUY executed for {self.ticker} at {self.entry_price:,.2f} KRW. TP: {self.r2:,.2f}, SL: {self.pp:,.2f}.")
                        else:
                            print(f"❌ BUY order failed or not closed for {self.ticker}.")
                    # 'breakout_down'은 선물 시장에서 매도 진입에 사용되므로, 현물에서는 거래 중지
                    elif current_price < (self.pp - self.breakout_value):
                        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 🔻 Breakout DOWN detected for {self.ticker}. Not trading in spot market.")
                        # 현물 시장에서는 매도 포지션 진입 불가, 따라서 아무것도 하지 않음
                else: # 포지션 보유 중: 익절 또는 손절 확인
                    # 익절 조건: 현재 가격이 R2에 도달
                    if current_price >= self.r2:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 🎉 Take Profit hit for {self.ticker} at {current_price:,.2f} KRW (R2: {self.r2:,.2f}).")
                        balances = self.upbit_service.get_balance()
                        amount_to_sell = balances['coins'].get(self.base_currency, 0)
                        if amount_to_sell > 0:
                            order = self.upbit_service.create_market_sell_order(self.ticker, amount_to_sell)
                            if order and order['status'] == 'closed':
                                print(f"🔴 SELL executed for {self.ticker} (Take Profit).")
                                self.position_held = False
                            else:
                                print(f"❌ SELL order failed or not closed for {self.ticker}.")
                        else:
                            print(f"Warning: No {self.base_currency} to sell for Take Profit.")
                        return # 전략 종료 (단일 거래 후 종료)

                    # 손절 조건: 현재 가격이 피봇 포인트 아래로 내려옴
                    elif current_price <= self.pp:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 🚨 Stop Loss hit for {self.ticker} at {current_price:,.2f} KRW (PP: {self.pp:,.2f}).")
                        balances = self.upbit_service.get_balance()
                        amount_to_sell = balances['coins'].get(self.base_currency, 0)
                        if amount_to_sell > 0:
                            order = self.upbit_service.create_market_sell_order(self.ticker, amount_to_sell)
                            if order and order['status'] == 'closed':
                                print(f"🔴 SELL executed for {self.ticker} (Stop Loss).")
                                self.position_held = False
                            else:
                                print(f"❌ SELL order failed or not closed for {self.ticker}.")
                        else:
                            print(f"Warning: No {self.base_currency} to sell for Stop Loss.")
                        return # 전략 종료 (단일 거래 후 종료)

            except Exception as e:
                print(f"An error occurred in BreakoutTrader run loop for {self.ticker}: {e}")
            
            time.sleep(interval_seconds)

if __name__ == '__main__':
    # 테스트를 위한 임시 .env 파일 생성 (필요시)
    import os
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("""UPBIT_ACCESS_KEY=YOUR_ACCESS_KEY
UPBIT_SECRET_KEY=YOUR_SECRET_KEY""")
        print(f"Created a dummy .env file at {env_path}. Please replace YOUR_ACCESS_KEY and UPBIT_SECRET_KEY with actual values.")
    load_dotenv(env_path)

    # BreakoutTrader 테스트 예시
    try:
        upbit_service = UpbitService()
        upbit_service.connect()

        ticker = 'BTC/KRW' # 예시 티커
        order_amount_krw = 100000 # 10만원

        breakout_trader = BreakoutTrader(upbit_service, ticker, order_amount_krw)
        # breakout_trader.run(interval_seconds=60) # 실제 실행 시 주석 해제
        print("BreakoutTrader example setup complete. To run, integrate into main.py and ensure API keys are set.")

    except Exception as e:
        print(f"An unexpected error occurred during BreakoutTrader setup: {e}")