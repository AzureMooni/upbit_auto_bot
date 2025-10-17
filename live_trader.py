
import pyupbit
import pandas as pd
import numpy as np
import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

from universe_manager import get_top_10_coins
from dl_predictor import predict_win_probability

# --- Configuration ---
load_dotenv(dotenv_path="config/.env")

UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY')

MODEL_PATH = "data/v2_lightgbm_model.joblib"
TRANSACTION_FEE = 0.0005
TRAILING_STOP_PCT = 0.10
WIN_PROB_THRESHOLD = 0.55
MAX_POSITION_RATIO = 0.25
HALF_KELLY_FACTOR = 0.5

NTFY_TOPIC = "upbit-live-trades"
LOG_FILE_PATH = "logs/live_trader.log"

class LiveTrader:
    def __init__(self):
        print("Initializing Live Trader...")
        self.upbit = self._connect_to_upbit()
        self.open_positions = {}
        self._load_initial_positions()

    def _connect_to_upbit(self):
        if not UPBIT_ACCESS_KEY or not UPBIT_SECRET_KEY:
            self.log("[FATAL] UPBIT_ACCESS_KEY or UPBIT_SECRET_KEY is not set in the .env file.")
            exit()
        try:
            upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
            balance = upbit.get_balance("KRW")
            if balance is None:
                self.log("[FATAL] Failed to fetch balance. Check API key permissions.")
                exit()
            self.log(f"[SUCCESS] Connected to Upbit. Current KRW Balance: {balance:,.0f} KRW")
            return upbit
        except Exception as e:
            self.log(f"[FATAL] Failed to connect to Upbit: {e}")
            exit()

    def _load_initial_positions(self):
        balances = self.upbit.get_balances()
        for balance in balances:
            ticker = f"KRW-{balance['currency']}"
            if balance['currency'] != 'KRW':
                amount = float(balance['balance'])
                avg_buy_price = float(balance['avg_buy_price'])
                current_price = pyupbit.get_current_price(ticker)
                self.open_positions[ticker] = {
                    'entry_price': avg_buy_price,
                    'peak_price': max(avg_buy_price, current_price if current_price else 0),
                    'amount': amount
                }
        if self.open_positions:
            self.log(f"[INFO] Loaded initial positions: {list(self.open_positions.keys())}")

    def log(self, message):
        log_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(log_msg)
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(log_msg + '\n')

    def send_notification(self, title, message):
        try:
            requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=message.encode('utf-8'), headers={"Title": title.encode('utf-8'), "Tags": "tada"})
        except Exception as e:
            self.log(f"[WARN] Failed to send ntfy notification: {e}")

    def check_exit_conditions(self):
        if not self.open_positions:
            return
        self.log("[INFO] Checking exit conditions for open positions...")
        for ticker in list(self.open_positions.keys()):
            try:
                position = self.open_positions[ticker]
                current_price = pyupbit.get_current_price(ticker)
                if current_price is None: continue

                position['peak_price'] = max(position['peak_price'], current_price)
                trailing_stop_price = position['peak_price'] * (1 - TRAILING_STOP_PCT)

                if current_price <= trailing_stop_price:
                    self.log(f"[EXIT] Trailing Stop triggered for {ticker} at {current_price:,.0f}")
                    sell_result = self.upbit.sell_market_order(ticker, position['amount'])
                    self.log(f"  - SELL order successful: {sell_result['uuid']}")
                    self.send_notification(f"✅ SELL: {ticker}", f"Price: {current_price:,.0f}, Amount: {position['amount']:.4f}")
                    del self.open_positions[ticker]
                time.sleep(0.2)
            except Exception as e:
                self.log(f"[ERROR] Error checking exit for {ticker}: {e}")

    def check_entry_conditions(self, universe):
        self.log("[INFO] Checking entry conditions for new positions...")
        try:
            krw_balance = self.upbit.get_balance("KRW")
            holdings_value = 0
            if self.open_positions:
                open_tickers = list(self.open_positions.keys())
                current_prices = pyupbit.get_current_price(open_tickers)
                
                # 보유 코인이 하나일 때와 여러 개일 때를 모두 처리
                if isinstance(current_prices, dict):
                    holdings_value = sum(p['amount'] * current_prices.get(t, 0) for t, p in self.open_positions.items())
                elif isinstance(current_prices, float) and len(open_tickers) == 1:
                    holdings_value = self.open_positions[open_tickers[0]]['amount'] * current_prices

            total_capital = krw_balance + holdings_value
        except Exception as e:
            self.log(f"[ERROR] Could not get account balance: {e}")
            return

        for ticker in universe:
            if ticker in self.open_positions: continue
            try:
                df = pyupbit.get_ohlcv(ticker, interval="minute60", count=80)
                if df is None or len(df) < 80: continue

                # [REVISED] Manual feature calculation
                features = df.copy()
                delta = features['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                features['RSI'] = 100 - (100 / (1 + (gain / loss)))
                ema_fast = features['close'].ewm(span=12, adjust=False).mean()
                ema_slow = features['close'].ewm(span=26, adjust=False).mean()
                features['MACD_hist'] = ema_fast - ema_slow - (ema_fast - ema_slow).ewm(span=9, adjust=False).mean()
                mid_band = features['close'].rolling(window=20).mean()
                std_dev = features['close'].rolling(window=20).std()
                features['BBP'] = (features['close'] - (mid_band - 2 * std_dev)) / (4 * std_dev)
                high_low = features['high'] - features['low']
                high_close = np.abs(features['high'] - features['close'].shift())
                low_close = np.abs(features['low'] - features['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                features['ATR'] = tr.rolling(window=14).mean()
                features.dropna(inplace=True)
                
                if features.empty: continue

                live_features = features.tail(1)[['RSI', 'MACD_hist', 'BBP', 'ATR']]
                p_win = predict_win_probability(live_features, MODEL_PATH)

                if p_win > WIN_PROB_THRESHOLD:
                    kelly_fraction = p_win - (1 - p_win)
                    position_size_ratio = min(kelly_fraction * HALF_KELLY_FACTOR, MAX_POSITION_RATIO)
                    position_size_krw = total_capital * position_size_ratio

                    if krw_balance >= position_size_krw and position_size_krw > 5000:
                        self.log(f"[ENTRY] BUY signal for {ticker} | P(win): {p_win:.2f} | Size: {position_size_krw:,.0f} KRW")
                        buy_result = self.upbit.buy_market_order(ticker, position_size_krw)
                        self.log(f"  - BUY order successful: {buy_result['uuid']}")
                        current_price = pyupbit.get_current_price(ticker)
                        bought_amount = position_size_krw / current_price
                        self.open_positions[ticker] = {'entry_price': current_price, 'peak_price': current_price, 'amount': bought_amount}
                        self.send_notification(f"✅ BUY: {ticker}", f"P(win): {p_win:.2f}, Size: {position_size_krw:,.0f} KRW")
                time.sleep(0.2)
            except Exception as e:
                self.log(f"[ERROR] Error checking entry for {ticker}: {e}")

    def run(self):
        """메인 실행 루프. 주기적으로 거래 로직을 실행합니다."""
        while True:
            self.log("\n--- Starting new trading cycle ---")
            
            # --- Bear Market Defense Protocol ---
            try:
                btc_df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=201)
                if btc_df is None or len(btc_df) < 200:
                    raise ValueError("Could not fetch sufficient BTC data for macro check.")
                
                btc_df['SMA_50'] = btc_df['close'].rolling(window=50).mean()
                btc_df['SMA_200'] = btc_df['close'].rolling(window=200).mean()
                
                latest_sma50 = btc_df['SMA_50'].iloc[-1]
                latest_sma200 = btc_df['SMA_200'].iloc[-1]

                if latest_sma50 < latest_sma200:
                    self.log("[DEFCON 1] BEAR MARKET DETECTED. Switching to Capital Preservation Mode.")
                    if self.open_positions:
                        for ticker in list(self.open_positions.keys()):
                            position = self.open_positions[ticker]
                            self.log(f"  - Liquidating {ticker}...")
                            self.upbit.sell_market_order(ticker, position['amount'])
                            self.send_notification(f"🚨 [DEFCON 1] Liquidated: {ticker}", "Bear market detected.")
                            del self.open_positions[ticker]
                    
                    self.log("--- Cycle finished (Capital Preservation). Waiting for 1 hour. ---")
                    time.sleep(3600)
                    return # Skip all other logic
            except Exception as e:
                self.log(f"[ERROR] CRITICAL: Failed to check macro market regime: {e}")
                self.log("--- Cycle finished due to critical error. Waiting for 1 hour. ---")
                time.sleep(3600)
                return
            # --- Protocol End ---

            # 1. 유니버스 업데이트
            universe = get_top_10_coins()
            # 2. 포지션 종료 조건 확인
            self.check_exit_conditions()
            # 3. 신규 진입 조건 확인
            self.check_entry_conditions(universe)
            
            self.log("--- Cycle finished. Waiting for 1 hour. ---")
            time.sleep(3600)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    trader = LiveTrader()
    trader.run()
