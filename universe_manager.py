import pandas as pd
from core.exchange import UpbitService # Import UpbitService

async def get_top_10_coins(upbit_service: UpbitService):
    """
    UpbitService를 사용하여 24시간 거래대금 기준으로 상위 10개 코인을 동적으로 선정하여 반환합니다.
    """
    print("[INFO] 동적 유니버스 선정을 시작합니다 (24시간 거래대금 기준 상위 10개).")
    
    try:
        all_markets = await upbit_service.exchange.fetch_markets()
        if not all_markets:
            print("[WARN] Upbit에서 티커 정보를 가져오지 못했습니다. 고정 유니버스를 반환합니다.")
            return [
                "BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW",
                "AVAX/KRW", "ADA/KRW", "LINK/KRW", "ETC/KRW", "TRX/KRW"
            ]

        volume_data = []
        for market in all_markets:
            symbol = market['symbol']
            if market['quote'] == 'KRW': # KRW 마켓만 고려
                try:
                    ticker_detail = await upbit_service.exchange.fetch_ticker(symbol)
                    if ticker_detail and 'quoteVolume' in ticker_detail:
                        volume_data.append({'symbol': symbol, 'volume': ticker_detail['quoteVolume']})
                except Exception:
                    # fetch_ticker에서 오류가 발생하는 일부 마켓(예: BCH-BTC)은 건너뜁니다.
                    continue
        
        # 거래량 기준으로 내림차순 정렬
        volume_df = pd.DataFrame(volume_data)
        if volume_df.empty:
            print("[WARN] 거래량 데이터를 가져오지 못했습니다. 고정 유니버스를 반환합니다.")
            return [
                "BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW",
                "AVAX/KRW", "ADA/KRW", "LINK/KRW", "ETC/KRW", "TRX/KRW"
            ]

        top_10_coins = volume_df.nlargest(10, 'volume')['symbol'].tolist()
        
        print(f"[SUCCESS] 동적 선정 유니버스 (상위 10개): {top_10_coins}")
        return top_10_coins

    except Exception as e:
        print(f"[ERROR] 동적 유니버스 선정 중 오류 발생: {e}. 고정 유니버스를 반환합니다.")
        return [
            "BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW",
            "AVAX/KRW", "ADA/KRW", "LINK/KRW", "ETC/KRW", "TRX/KRW"
        ]

if __name__ == '__main__':
    # 이 부분은 UpbitService 인스턴스가 필요하므로 직접 실행하려면 비동기 환경 설정이 필요합니다.
    # 예시를 위해 임시 UpbitService 객체를 생성하거나, 비동기 함수 내에서 호출해야 합니다.
    print("universe_manager.py는 직접 실행 시 UpbitService 인스턴스가 필요합니다.")
    print("LiveTrader 내에서 호출될 때 정상 작동합니다.")