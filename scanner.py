import ccxt
import pandas as pd
import time
import pandas_ta as ta
import os
import numpy as np
import asyncio
from core.exchange import UpbitService
from dl_model_trainer import DLModelTrainer

async def _get_ohlcv_df(exchange: ccxt.Exchange, ticker: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetches OHLCV data from ccxt and converts it to a pandas DataFrame.
    """
    try:
        ohlcv = await exchange.fetch_ohlcv(ticker, timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching OHLCV for {ticker} ({timeframe}): {e}")
        return pd.DataFrame()

def _calculate_all_indicators(df: pd.DataFrame, ema_short_period: int = 30, ema_long_period: int = 100, adx_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    """
    Calculates all necessary technical indicators on the given DataFrame.
    Assumes df is 1-hour OHLCV data.
    """
    if df.empty:
        return df

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

    df_4h = df['close'].resample('4h').ohlc().dropna()
    if not df_4h.empty:
        df_4h[f'EMA_{ema_short_period}'] = df_4h['close'].ta.ema(length=ema_short_period)
        df_4h[f'EMA_{ema_long_period}'] = df_4h['close'].ta.ema(length=ema_long_period)
        df = df.merge(df_4h[[f'EMA_{ema_short_period}', f'EMA_{ema_long_period}']], left_index=True, right_index=True, how='left')
        df[f'EMA_{ema_short_period}'] = df[f'EMA_{ema_short_period}'].ffill()
        df[f'EMA_{ema_long_period}'] = df[f'EMA_{ema_long_period}'].ffill()

    df.ta.rsi(length=14, append=True, close='close')
    df.ta.bbands(length=20, std=2, append=True, close='close')
    df.ta.adx(length=adx_period, append=True, high='high', low='low', close='close')
    df.ta.atr(length=atr_period, append=True, high='high', low='low', close='close')

    df_daily = df['close'].resample('1D').ohlc().dropna()
    if len(df_daily) >= 2:
        prev_day = df_daily.iloc[-2]
        pp = (prev_day['high'] + prev_day['low'] + prev_day['close']) / 3
        k = 0.5
        breakout_value = (prev_day['high'] - prev_day['low']) * k
        df['PP'] = pp
        df['BREAKOUT_VALUE'] = breakout_value
        df['PP'] = df['PP'].ffill()
        df['BREAKOUT_VALUE'] = df['BREAKOUT_VALUE'].ffill()

    return df

def find_hot_coin(df_1h: pd.DataFrame, dl_trainer: DLModelTrainer, market_regime: str) -> list:
    """
    Identifies the "hottest" coin from a given DataFrame based on DL model predictions.
    This is the "backtest" version.
    """
    if market_regime == 'Bullish':
        BUY_PROBABILITY_THRESHOLD = 0.55
    elif market_regime == 'Bearish':
        BUY_PROBABILITY_THRESHOLD = 0.75
    else:  # Sideways
        BUY_PROBABILITY_THRESHOLD = 0.65
    
    print(f"🔥 DL 모델로 핫 코인 스캔 중 (백테스트 모드) | 시장: {market_regime}, 임계값: {BUY_PROBABILITY_THRESHOLD}")

    if dl_trainer.model is None:
        print("DL 모델이 로드되지 않았습니다.")
        return []

    # 데이터 파이프라인이 특징 생성을 처리하므로 원본 df를 그대로 전달
    probabilities = dl_trainer.predict_proba(df_1h.copy())
    if probabilities is not None:
        buy_probability = probabilities[1]
        if buy_probability >= BUY_PROBABILITY_THRESHOLD:
            print(f"🏆 핫 코인 발견! (매수 확률: {buy_probability:.4f})")
            return ["DUMMY/KRW"] # 백테스트용 더미 티커 반환
    return []

async def find_hot_coin_live(exchange: ccxt.Exchange, dl_trainer: DLModelTrainer, market_regime: str) -> list:
    """
    실시간으로 모든 타겟 코인의 데이터를 가져와 딥러닝 모델의 예측 확률에 기반하여
    가장 "뜨거운" 코인을 식별합니다. 가장 좋은 코인 하나를 리스트에 담아 반환합니다.
    """
    if market_regime == 'Bullish':
        BUY_PROBABILITY_THRESHOLD = 0.55
    elif market_regime == 'Bearish':
        BUY_PROBABILITY_THRESHOLD = 0.75
    else:  # Sideways
        BUY_PROBABILITY_THRESHOLD = 0.65

    print(f"🔥 DL 모델로 핫 코인 스캔 중 (라이브 모드) | 시장: {market_regime}, 임계값: {BUY_PROBABILITY_THRESHOLD}")

    if dl_trainer.model is None:
        print("DL 모델이 로드되지 않아 핫 코인을 찾을 수 없습니다.")
        return []

    for ticker in DLModelTrainer.TARGET_COINS:
        df_1h = await _get_ohlcv_df(exchange, ticker, '1h', limit=dl_trainer.sequence_length + 100)

        if df_1h is None or df_1h.empty or len(df_1h) < dl_trainer.sequence_length:
            print(f"[{ticker}] 데이터가 예측에 충분하지 않습니다.")
            continue

        # 데이터 파이프라인이 특징 생성을 처리하므로 원본 df를 그대로 전달
        probabilities = dl_trainer.predict_proba(df_1h.copy())

        if probabilities is not None:
            buy_probability = probabilities[1]
            print(f"  - {ticker} | 매수 확률: {buy_probability:.4f}")
            if buy_probability >= BUY_PROBABILITY_THRESHOLD:
                hot_coin_candidates.append((ticker, buy_probability))
    
    if not hot_coin_candidates:
        print("기준에 맞는 핫 코인을 찾지 못했습니다.")
        return []

    hot_coin_candidates.sort(key=lambda x: x[1], reverse=True)
    best_coin = hot_coin_candidates[0]
    print(f"🏆 가장 유력한 핫 코인: {best_coin[0]} (매수 확률: {best_coin[1]:.4f})")
    
    return [best_coin[0]]

def get_dynamic_grid_prices(df_1h: pd.DataFrame):
    """
    Calculates dynamic grid prices using Bollinger Bands from a 1-hour OHLCV DataFrame.
    """
    if df_1h.empty or len(df_1h) < 20:
        print(f"Not enough OHLCV data to calculate Bollinger Bands. (Need 20, got {len(df_1h)})")
        return None, None
    
    try:
        df_with_indicators = _calculate_all_indicators(df_1h.copy())
        
        upper_price = df_with_indicators['BBU_20_2.0'].iloc[-1]
        lower_price = df_with_indicators['BBL_20_2.0'].iloc[-1]
        
        print(f"Dynamic Grid Prices: Lower Band = {lower_price:.2f}, Upper Band = {upper_price:.2f}")
        return lower_price, upper_price

    except Exception as e:
        print(f"Error in get_dynamic_grid_prices: {e}")
        return None, None

async def get_dynamic_grid_prices_live(upbit_exchange: ccxt.Exchange, ticker: str):
    """
    Fetches live 1-hour OHLCV data and calculates dynamic grid prices using Bollinger Bands.
    """
    df_1h = await _get_ohlcv_df(upbit_exchange, ticker, '1h', limit=20)
    if df_1h.empty:
        return None, None
    
    return get_dynamic_grid_prices(df_1h)

def classify_market(df_1h: pd.DataFrame):
    """
    Classifies market type based on a 1-hour OHLCV DataFrame,
    prioritizing breakout conditions, then ADX.
    """
    if df_1h.empty or len(df_1h) < 100:
        print(f"Not enough OHLCV data to classify market. (Need at least 100, got {len(df_1h)})")
        return "unknown"

    try:
        df_with_indicators = _calculate_all_indicators(df_1h.copy())
        current_price = df_1h['close'].iloc[-1]

        if 'PP' in df_with_indicators.columns and 'BREAKOUT_VALUE' in df_with_indicators.columns:
            pp = df_with_indicators['PP'].iloc[-1]
            breakout_value = df_with_indicators['BREAKOUT_VALUE'].iloc[-1]
            if current_price > (pp + breakout_value):
                return "breakout_up"
            elif current_price < (pp - breakout_value):
                return "breakout_down"

        if 'ADX_14' in df_with_indicators.columns:
            latest_adx = df_with_indicators['ADX_14'].iloc[-1]
            if latest_adx >= 25:
                return "trending"
            elif latest_adx < 20:
                return "ranging"
            else:
                return "choppy"
        
        return "unknown"

    except Exception as e:
        print(f"Error in classify_market: {e}")
        return "unknown"

async def classify_market_live(upbit_exchange: ccxt.Exchange, ticker: str):
    """
    Fetches live 1-hour OHLCV data and classifies market type.
    """
    df_1h = await _get_ohlcv_df(upbit_exchange, ticker, '1h', limit=150)
    if df_1h.empty:
        return "unknown"
    
    return classify_market(df_1h)

def _calculate_breakout_levels_from_df(df_daily: pd.DataFrame):
    if len(df_daily) < 2:
        return None, None, None, None, None, None

    prev_day = df_daily.iloc[-2]
    prev_high = prev_day['high']
    prev_low = prev_day['low']
    prev_close = prev_day['close']

    pp = (prev_high + prev_low + prev_close) / 3
    r1 = (2 * pp) - prev_low
    s1 = (2 * pp) - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    k = 0.5
    breakout_value = (prev_high - prev_low) * k

    return pp, r1, s1, r2, s2, breakout_value

if __name__ == '__main__':
    print("--- scanner.py 기능 테스트 ---")

    dummy_ohlcv_btc = [
        [1672531200000, 20000000, 20100000, 19900000, 20050000, 1000],
        *[ 
            [1672531200000 + (i * 3600000), 20050000 + (i*10000), 20200000 + (i*10000), 20000000 + (i*10000), 20150000 + (i*10000), 1200 + i] 
            for i in range(1, 201)
        ]
    ]
    dummy_df_btc = pd.DataFrame(dummy_ohlcv_btc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    dummy_df_btc['timestamp'] = pd.to_datetime(dummy_df_btc['timestamp'], unit='ms')
    dummy_df_btc.set_index('timestamp', inplace=True)

    dummy_historical_data = {
        'BTC/KRW': dummy_df_btc,
    }

    async def test_scanner_functions():
        dl_trainer = DLModelTrainer()
        dl_trainer.load_model()

        print("\n--- 라이브 모드 테스트 ---")
        upbit_service = UpbitService()
        await upbit_service.connect()
        live_exchange = upbit_service.exchange

        print("\n핫 코인 찾기 (라이브)...")
        hot_coins_live = await find_hot_coin_live(live_exchange, dl_trainer, market_regime='Sideways')
        if hot_coins_live:
            print(f"찾은 핫 코인 (라이브): {hot_coins_live}")
            first_hot_coin_live = hot_coins_live[0]
            
            print(f"\n{first_hot_coin_live}의 동적 그리드 가격 가져오기 (라이브)...")
            lower_live, upper_live = await get_dynamic_grid_prices_live(live_exchange, first_hot_coin_live)
            if lower_live and upper_live:
                print(f"동적 그리드 범위 (라이브): Lower={lower_live:.2f}, Upper={upper_live:.2f}")
        else:
            print("라이브 모드에서 핫 코인을 찾지 못했습니다.")

        print("\nBTC/KRW 시장 유형 분류 (라이브)...")
        market_type_live = await classify_market_live(live_exchange, 'BTC/KRW')
        print(f"BTC/KRW 시장 유형 (라이브): {market_type_live}")

        print("\n--- 백테스트 모드 테스트 ---")

        print("\n핫 코인 찾기 (백테스트)...")
        hot_coins_backtest = find_hot_coin(df_1h=dummy_historical_data['BTC/KRW'], dl_trainer=dl_trainer, market_regime='Sideways')
        if hot_coins_backtest:
            print(f"찾은 핫 코인 (백테스트): {hot_coins_backtest}")
            first_hot_coin_backtest = hot_coins_backtest[0]
            
            print(f"\n{first_hot_coin_backtest}의 동적 그리드 가격 가져오기 (백테스트)...")
            lower_backtest, upper_backtest = get_dynamic_grid_prices(df_1h=dummy_historical_data['BTC/KRW'])
            if lower_backtest and upper_backtest:
                print(f"동적 그리드 범위 (백테스트): Lower={lower_backtest:.2f}, Upper={upper_backtest:.2f}")
        else:
            print("백테스트 모드에서 핫 코인을 찾지 못했습니다.")

        print("\nBTC/KRW 시장 유형 분류 (백테스트)...")
        market_type_backtest = classify_market(df_1h=dummy_historical_data['BTC/KRW'])
        print(f"BTC/KRW 시장 유형 (백테스트): {market_type_backtest}")

    asyncio.run(test_scanner_functions())