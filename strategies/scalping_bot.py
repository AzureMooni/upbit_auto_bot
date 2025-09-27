import time
import pandas as pd
import pandas_ta as ta
import ccxt
from core.exchange import UpbitService

class ScalpingBot:
    def __init__(self, upbit_service: UpbitService, ticker: str, allocated_capital: float):
        self.upbit_service = upbit_service
        self.ticker = ticker
        self.allocated_capital = allocated_capital
        self.position_held = False
        self.entry_price = 0.0
        self.base_currency = ticker.split('/')[0]
        self.take_profit_ratio = 1.02 # +2% 익절
        self.stop_loss_ratio = 0.99 # -1% 손절

        print(f"ScalpingBot initialized for {self.ticker} with TP: +{self.take_profit_ratio-1:.0%}, SL: -{1-self.stop_loss_ratio:.0%}. Allocated capital: {self.allocated_capital:,.0f} KRW.")

    def _get_ohlcv(self, timeframe='15m', limit=20):
        """
        지정된 타임프레임과 리밋으로 OHLCV 데이터를 가져옵니다.
        """
        try:
            ohlcv = self.upbit_service.exchange.fetch_ohlcv(self.ticker, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < limit:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {self.ticker}: {e}")
            return None

    def run(self, interval_seconds: int = 15):
        """
        스캘핑 전략을 실행합니다.
        """
        print(f"Starting ScalpingBot for {self.ticker}...")
        while True:
            try:
                df = self._get_ohlcv(timeframe='15m', limit=20)
                if df is None:
                    print(f"Not enough data for {self.ticker}. Retrying...")
                    time.sleep(interval_seconds)
                    continue

                # EMA 계산
                df.ta.ema(length=5, append=True, close='close')
                df.ta.ema(length=10, append=True, close='close')

                ema5 = df['EMA_5'].iloc[-1]
                ema10 = df['EMA_10'].iloc[-1]
                prev_ema5 = df['EMA_5'].iloc[-2]
                prev_ema10 = df['EMA_10'].iloc[-2]

                current_price = self.upbit_service.get_current_price(self.ticker)
                if current_price is None:
                    print(f"Could not fetch current price for {self.ticker}. Retrying...")
                    time.sleep(interval_seconds)
                    continue

                if not self.position_held:
                    # 진입 신호: 5 EMA가 10 EMA를 상향 돌파 (골든 크로스)
                    if ema5 > ema10 and prev_ema5 <= prev_ema10:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 📈 Golden Cross detected for {self.ticker}. Attempting to BUY.")
                        order = self.upbit_service.create_market_buy_order(self.ticker, self.allocated_capital)
                        if order and order['status'] == 'closed':
                            self.position_held = True
                            self.entry_price = current_price
                            print(f"🟢 BUY executed for {self.ticker} at {self.entry_price:,.2f} KRW.")
                        else:
                            print(f"❌ BUY order failed or not closed for {self.ticker}.")
                else:
                    # 포지션 보유 중: 익절 또는 손절 확인
                    take_profit_price = self.entry_price * self.take_profit_ratio
                    stop_loss_price = self.entry_price * self.stop_loss_ratio

                    if current_price >= take_profit_price:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 🎉 Take Profit hit for {self.ticker} at {current_price:,.2f} KRW.")
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

                    elif current_price <= stop_loss_price:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] 🚨 Stop Loss hit for {self.ticker} at {current_price:,.2f} KRW.")
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
                print(f"An error occurred in ScalpingBot run loop for {self.ticker}: {e}")
            
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

    # ScalpingBot 테스트 예시
    try:
        upbit_service = UpbitService()
        upbit_service.connect()

        # 예시 파라미터
        ticker = 'BTC/KRW'
        order_amount_krw = 100000 # 10만원

        scalping_bot = ScalpingBot(upbit_service, ticker, order_amount_krw)
        # scalping_bot.run(interval_seconds=15) # 실제 실행 시 주석 해제
        print("ScalpingBot example setup complete. To run, uncomment 'scalping_bot.run()' and ensure API keys are set.")

    except Exception as e:
        print(f"An unexpected error occurred during ScalpingBot setup: {e}")
