
import pyupbit
import os
from dotenv import load_dotenv

# .env 파일 로드 (루트 디렉토리 기준)
load_dotenv()

access_key = os.getenv('UPBIT_ACCESS_KEY')
secret_key = os.getenv('UPBIT_SECRET_KEY')

print("--- API Key & Permission Test ---")

if not access_key or not secret_key:
    print("[FAIL] API keys not found in config/.env file.")
    exit()

print("[INFO] API keys loaded successfully.")

try:
    upbit = pyupbit.Upbit(access_key, secret_key)
    krw_balance = upbit.get_balance("KRW")
    
    if krw_balance is not None:
        print(f"[SUCCESS] Successfully connected and fetched balance.")
        print(f"  - KRW Balance: {krw_balance:,.0f} KRW")
    else:
        print("[FAIL] Connection successful, but failed to fetch balance. Please re-check API key 'Asset Inquiry' permission.")

except Exception as e:
    print(f"[FAIL] An error occurred during API connection: {e}")
