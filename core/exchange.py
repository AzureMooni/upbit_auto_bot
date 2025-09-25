import os
from dotenv import load_dotenv
import ccxt

class UpbitService:
    def __init__(self):
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))
        self.access_key = os.getenv('UPBIT_ACCESS_KEY')
        self.secret_key = os.getenv('UPBIT_SECRET_KEY')
        self.exchange = None

        if not self.access_key or not self.secret_key:
            raise ValueError("UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY must be set in config/.env")

    def connect(self):
        """
        ccxt 라이브러리를 사용하여 업비트 거래소 객체를 초기화하고 인증을 완료합니다.
        """
        self.exchange = ccxt.upbit({
            'apiKey': self.access_key,
            'secret': self.secret_key,
            'options': {
                'defaultType': 'spot',
            },
        })
        print("Upbit exchange connected successfully.")
        return self.exchange

    def get_balance(self):
        """
        현재 계정의 전체 자산 잔고(KRW 및 보유 코인)를 조회하여 딕셔너리 형태로 반환합니다.
        """
        if not self.exchange:
            raise ConnectionError("Exchange not connected. Call connect() first.")
        
        balances = self.exchange.fetch_balance()
        
        # KRW 잔고
        krw_balance = balances['free']['KRW'] if 'KRW' in balances['free'] else 0
        
        # 보유 코인 잔고
        coin_balances = {}
        for currency, balance_info in balances['free'].items():
            if currency != 'KRW' and balance_info > 0:
                coin_balances[currency] = balance_info
        
        return {
            'KRW': krw_balance,
            'coins': coin_balances
        }

    def get_current_price(self, ticker: str):
        """
        특정 티커(예: 'BTC/KRW')의 현재 가격을 조회하여 숫자(float)로 반환합니다.
        """
        if not self.exchange:
            raise ConnectionError("Exchange not connected. Call connect() first.")
        
        try:
            ticker_data = self.exchange.fetch_ticker(ticker)
            return float(ticker_data['last'])
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            return None

if __name__ == '__main__':
    # .env 파일 생성 (테스트용)
    env_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("UPBIT_ACCESS_KEY=YOUR_ACCESS_KEY
")
            f.write("UPBIT_SECRET_KEY=YOUR_SECRET_KEY
")
        print(f"Created a dummy .env file at {env_path}. Please replace YOUR_ACCESS_KEY and YOUR_SECRET_KEY with actual values.")

    # 사용 예시 (실제 API 키 필요)
    try:
        upbit_service = UpbitService()
        upbit_service.connect()

        # 잔고 조회
        balance = upbit_service.get_balance()
        print(f"Current Balance: {balance}")

        # 특정 코인 가격 조회
        btc_price = upbit_service.get_current_price('BTC/KRW')
        if btc_price:
            print(f"Current BTC/KRW Price: {btc_price}")
        
        eth_price = upbit_service.get_current_price('ETH/KRW')
        if eth_price:
            print(f"Current ETH/KRW Price: {eth_price}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except ConnectionError as e:
        print(f"Connection Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
