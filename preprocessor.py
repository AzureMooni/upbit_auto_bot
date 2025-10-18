import pandas as pd
import os
from market_regime_detector import precompute_all_indicators, get_market_regime
from strategies.trend_follower import generate_v_recovery_signals
from strategies.mean_reversion_strategy import generate_sideways_signals

def preprocess_data(file_path: str, output_path: str):
    """
    RL 훈련을 위해 모든 지표와 시장 체제를 계산하여 전처리된 데이터를 생성합니다.
    """
    print(f"데이터를 로드합니다: {file_path}")
    if not os.path.exists(file_path):
        print(f"오류: 데이터 파일이 없습니다 - {file_path}")
        return

    df = pd.read_feather(file_path).set_index("timestamp")
    
    print("모든 기술적 지표와 시장 체제를 계산합니다...")
    # 1. 시장 체제 분석에 필요한 모든 지표 계산
    df_processed = precompute_all_indicators(df)

    # 2. V-회복 및 횡보장 전략 신호에 필요한 지표 추가 계산
    df_processed = generate_v_recovery_signals(df_processed)
    df_processed = generate_sideways_signals(df_processed)

    # 3. 시장 체제 자체를 피처로 추가 (숫자로 변환)
    daily_regime = df_processed.apply(get_market_regime, axis=1).rename('regime')
    df_processed['regime'] = daily_regime.reindex(df_processed.index.date).set_axis(df_processed.index)
    df_processed['regime'] = df_processed['regime'].ffill()

    regime_map = {name: i for i, name in enumerate(df_processed['regime'].unique())}
    df_processed['regime'] = df_processed['regime'].map(regime_map)

    # 4. RL 환경에 필요한 최종 피처 선택
    # OHLCV (5) + ADX, Norm_ATR, BBP, EMA_20, EMA_50, RSI, MACD_hist, regime (8) = 13 features
    final_features = [
        'open', 'high', 'low', 'close', 'volume',
        'ADX', 'Normalized_ATR', 'BBP', 'EMA_20', 'EMA_50',
        'RSI_14', 'MACD_hist', 'regime'
    ]
    df_final = df_processed[final_features].dropna()

    print(f"전처리가 완료되었습니다. {len(df_final)}개의 데이터 포인트를 저장합니다.")
    df_final.to_pickle(output_path)
    print(f"전처리된 데이터가 다음 경로에 저장되었습니다: {output_path}")

if __name__ == '__main__':
    # 데이터 캐시 디렉토리 확인 및 생성
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    preprocess_data(
        file_path=os.path.join(cache_dir, 'BTC_KRW_1h.feather'),
        output_path=os.path.join(cache_dir, 'preprocessed_data.pkl')
    )
