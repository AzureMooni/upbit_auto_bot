

import asyncio
import pandas as pd
from core.exchange import UpbitService
from dl_model_trainer import DLModelTrainer

# --- Helper Functions (Pure Logic) ---

def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """주어진 DataFrame에 모든 기술적 지표를 계산하여 추가합니다."""
    if df.empty:
        return df
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    return df

def _find_best_coin(candidates: list) -> tuple | None:
    """매수 확률이 가장 높은 코인을 찾습니다."""
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]

# --- Main Scanner Functions ---

async def scan_for_hot_coin(dl_trainer: DLModelTrainer, market_regime: str, upbit_service: UpbitService) -> str | None:
    """
    실시간으로 여러 코인을 스캔하여 현재 가장 투자 매력도가 높은 코인(핫 코인)을 찾습니다.
    """
    if market_regime == 'Bullish':
        threshold = 0.55
    elif market_regime == 'Bearish':
        threshold = 0.75
    else:  # Sideways
        threshold = 0.65

    print(f"🔥 핫 코인 스캔 시작 (시장: {market_regime}, 매수 임계값: {threshold:.2f})")

    if dl_trainer.model is None:
        print("경고: DL 모델이 로드되지 않아 핫 코인 스캔을 건너뜁니다.")
        return None

    async def get_prediction(ticker: str):
        df = await upbit_service.get_ohlcv(ticker, '1h', limit=dl_trainer.sequence_length + 5)
        if df is None or len(df) < dl_trainer.sequence_length:
            return None
        
        proba = dl_trainer.predict_proba(df.copy())
        if proba is not None:
            buy_proba = proba[1]
            print(f"  - {ticker} | 매수 확률: {buy_proba:.4f}")
            if buy_proba >= threshold:
                return ticker, buy_proba
        return None

    tasks = [get_prediction(ticker) for ticker in DLModelTrainer.TARGET_COINS]
    results = await asyncio.gather(*tasks)
    
    hot_coin_candidates = [res for res in results if res is not None]

    best_coin_info = _find_best_coin(hot_coin_candidates)

    if best_coin_info:
        print(f"🏆 핫 코인 발견: {best_coin_info[0]} (매수 확률: {best_coin_info[1]:.4f})")
        return best_coin_info[0]
    else:
        print("기준을 만족하는 핫 코인을 찾지 못했습니다.")
        return None

def calculate_grid_prices(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """
    DataFrame을 받아 볼린저 밴드를 기반으로 동적 그리드 가격을 계산합니다.
    """
    if df is None or len(df) < 20:
        return None, None
    
    df_with_indicators = _calculate_indicators(df.copy())
    
    upper_band = df_with_indicators['BBU_20_2.0'].iloc[-1]
    lower_band = df_with_indicators['BBL_20_2.0'].iloc[-1]
    
    return lower_band, upper_band

# --- Example Usage ---
async def main():
    print("--- scanner.py 기능 테스트 ---")
    upbit_service = None
    try:
        # 서비스 초기화
        upbit_service = UpbitService()
        await upbit_service.connect() 
        
        # DL 모델 로드 (실제로는 외부에서 주입받아야 함)
        dl_trainer = DLModelTrainer()
        dl_trainer.load_model()

        # 1. 핫 코인 스캔 테스트
        hot_coin = await scan_for_hot_coin(dl_trainer, 'Sideways', upbit_service)
        if hot_coin:
            # 2. 그리드 가격 계산 테스트
            df_hot_coin = await upbit_service.get_ohlcv(hot_coin, '1h', 20)
            lower, upper = calculate_grid_prices(df_hot_coin)
            if lower and upper:
                print(f"\n{hot_coin}의 동적 그리드 가격:")
                print(f"  - 상단: {upper:.2f}")
                print(f"  - 하단: {lower:.2f}")

    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
    finally:
        if upbit_service:
            await upbit_service.close()

if __name__ == '__main__':
    asyncio.run(main())
