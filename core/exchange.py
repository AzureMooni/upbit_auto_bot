import ccxt.async_support as ccxt
import asyncio
import pandas as pd

class UpbitService:
    """ Manages all API interactions with the Upbit exchange using ccxt. """
    def __init__(self, access_key: str, secret_key: str):
        self.exchange = ccxt.upbit({
            'apiKey': access_key,
            'secret': secret_key,
            'enableRateLimit': True,
        })
        print("Upbit exchange connected successfully.")

    async def connect(self):
        try:
            # Test connection by fetching markets (a lightweight call)
            await self.exchange.load_markets()
            print("✅ Upbit 주문 실행 엔진 활성화.")
        except Exception as e:
            print(f"[FATAL] Upbit 연결 실패: {e}")
            await self.close()
            raise

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            print("Upbit exchange connection closed.")

    async def get_balance(self, currency: str):
        try:
            balances = await self.exchange.fetch_balance()
            return balances.get(currency, {}).get('free', 0)
        except Exception as e:
            print(f"[ERROR] 잔고 조회 실패: {e}")
            return None

    async def get_all_balances(self):
        try:
            balances = await self.exchange.fetch_balance()
            # 'free' 밸런스가 있는 모든 자산 반환
            return {
                ticker: {'balance': info['free']}
                for ticker, info in balances['free'].items() if info > 0
            }
        except Exception as e:
            print(f"[ERROR] 전체 잔고 조회 실패: {e}")
            return None

    async def get_current_price(self, ticker: str):
        try:
            return (await self.exchange.fetch_ticker(ticker))['last']
        except Exception:
            return None

    async def get_ohlcv(self, ticker: str, timeframe='1h', limit=200):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"[ERROR] {ticker} OHLCV 데이터 가져오기 실패: {e}")
            return None