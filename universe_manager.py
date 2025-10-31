
def get_top_10_coins():
    """
    백테스트를 위해 유동성이 높은 대표 코인 10개로 구성된 고정 유니버스를 반환합니다.
    """
    print("[INFO] Returning a fixed, representative universe for backtesting.")
    
    # 유동성과 대표성을 고려한 고정 유니버스
    fixed_universe = [
        "KRW-BTC",
        "KRW-ETH",
    ]
    
    print(f"[SUCCESS] Selected Fixed Universe: {fixed_universe}")
    return fixed_universe

if __name__ == '__main__':
    trading_universe = get_top_10_coins()
    print("\n--- Trading Universe ---")
    print(trading_universe)
