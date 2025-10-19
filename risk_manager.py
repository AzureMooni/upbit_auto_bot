import pandas as pd

class RiskManager:
    def calculate_kelly_fraction(self, win_rate: float, avg_profit: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 0.0
        R = avg_profit / avg_loss
        if R == 0:
            return 0.0
        W = win_rate
        kelly_fraction = W - ((1 - W) / R)
        return kelly_fraction

def get_position_size_ratio(regime: str, normalized_atr: float, natr_ma: float) -> float:
    """
    시장 체제와 변동성을 함께 고려하여 동적으로 투자 비율을 결정합니다.

    Args:
        regime (str): 현재 시장 체제.
        normalized_atr (float): 현재의 Normalized ATR 값.
        natr_ma (float): Normalized ATR의 이동평균 값.

    Returns:
        float: 최종 투자 비율.
    """
    # 1. 시장 체제에 따른 기본 투자 비율 설정
    allocation_map = {
        'SIDEWAYS_LOW_VOL': 0.50,
        'SIDEWAYS_HIGH_VOL': 0.25,
        'TREND_LOW_VOL': 0.40,
        'TREND_HIGH_VOL': 0.20,
        'UNDEFINED': 0.0
    }
    base_ratio = allocation_map.get(regime, 0.0)

    if base_ratio == 0.0 or pd.isna(normalized_atr) or pd.isna(natr_ma) or natr_ma == 0:
        return 0.0

    # 2. 변동성 계수(volatility factor) 계산
    # 현재 변동성이 평균 대비 얼마나 높은지 측정. 1보다 크면 평소보다 변동성이 높다는 의미.
    volatility_factor = normalized_atr / natr_ma
    
    # 변동성 계수가 너무 극단적으로 변하지 않도록 상/하한 설정 (예: 0.5배 ~ 2배)
    volatility_factor = max(0.5, min(volatility_factor, 2.0))

    # 3. 최종 투자 비율 계산: 변동성이 높으면 투자 비율을 줄임 (역수 적용)
    final_ratio = base_ratio / volatility_factor

    return final_ratio

def get_position_size(regime: str, capital: float) -> float:
    """
    Determines the position size as an amount of capital based on the market regime.

    Args:
        regime (str): The current market regime string.
        capital (float): The total available trading capital.

    Returns:
        float: The amount of capital to be used for the next trade.
    """
    allocation_ratio = get_position_size_ratio(regime)
    return capital * allocation_ratio

# Example Usage:
if __name__ == '__main__':
    total_capital = 10000  # Example: $10,000

    regimes = [
        'SIDEWAYS_LOW_VOL',
        'SIDEWAYS_HIGH_VOL',
        'TREND_LOW_VOL',
        'TREND_HIGH_VOL',
        'UNDEFINED',
        'INVALID_REGIME' # Test default case
    ]

    for r in regimes:
        position_ratio = get_position_size_ratio(r)
        position_size = get_position_size(r, total_capital)
        print(f"Regime: {r:<20} | Ratio: {position_ratio:.2f} | Position Size: ${position_size:,.2f}")