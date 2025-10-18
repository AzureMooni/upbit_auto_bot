import sys
import pyupbit
import traceback
from universe_manager import get_top_10_coins

def main():
    print("--- [DEBUG] Initializing Live Trader v3 ---")

    # 1. API 키를 명령줄 인수에서 가져옵니다.
    if len(sys.argv) != 3:
        print("[FATAL] API Keys were not provided as command-line arguments.")
        sys.exit(1)

    access_key = sys.argv[1]
    secret_key = sys.argv[2]

    print(f"[DEBUG] Access Key loaded, starts with: {access_key[:4]}...")

    # 2. 'try...except' 없이 바로 접속을 시도합니다.
    # 이렇게 하면 오류 발생 시 프로그램이 즉시 'Traceback'을 출력하고 멈춥니다.
    print("[DEBUG] Attempting to create Upbit client...")
    upbit = pyupbit.Upbit(access_key, secret_key)

    print("[DEBUG] Attempting to fetch balance...")
    balance = upbit.get_balance("KRW")

    print(f"[SUCCESS] Balance check successful! KRW: {balance}")

    # 성공 시, 실제 봇의 메인 로직을 여기서 호출
    # run_my_bot_logic(upbit) 

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 만약 main() 함수 내부에서 예외가 발생하면,
        # 여기서 상세한 Traceback을 출력합니다.
        print(f"[FATAL] An unhandled exception occurred in main:")
        print(traceback.format_exc())
        sys.exit(1)