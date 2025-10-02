import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

class SentimentAnalyzer:
    """
    Gemini API를 사용하여 특정 암호화폐에 대한 시장 감성을 분석하고, 
    이를 -1.0(극단적 공포)에서 +1.0(극단적 탐욕) 사이의 점수로 계량화합니다.
    """
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-pro-latest')

    def get_sentiment_score(self, ticker: str) -> tuple[float, str]:
        """
        주어진 티커에 대한 시장 감성 점수와 분석 근거를 반환합니다.

        Args:
            ticker (str): 분석할 암호화폐 티커 (예: 'BTC/KRW')

        Returns:
            tuple[float, str]: (감성 점수, 분석 근거 요약). 오류 발생 시 (0.0, "분석 실패")
        """
        print(f"  - [Sentiment] '{ticker}' 시장 감성 분석 중...")
        try:
            prompt = f"""
            당신은 최고의 암호화폐 퀀트 분석가입니다.
            현재 '{ticker}' 암호화폐 시장의 전반적인 투자 심리를 분석해주세요.
            최신 뉴스, 기술적 지표(RSI, MACD 등), 온체인 데이터, 소셜 미디어 동향, 커뮤니티 여론을 종합적으로 고려해야 합니다.
            
            분석 결과를 바탕으로, 현재 시장의 심리를 -1.0(극단적 공포)에서 +1.0(극단적 탐욕) 사이의 점수로 평가하고, 그 점수를 부여한 핵심 근거를 1~2문장으로 요약해주세요.
            
            출력 형식은 반드시 아래와 같이 "점수: [숫자]" 와 "근거: [요약]" 형식으로 작성해주세요.
            
            점수: [여기에 -1.0에서 +1.0 사이의 숫자만 삽입]
            근거: [여기에 핵심 근거 요약 삽입]
            """
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # 점수와 근거를 파싱
            score_match = re.search(r"점수:\s*(-?\d+\.?\d*)", response_text)
            summary_match = re.search(r"근거:\s*(.*)", response_text, re.DOTALL)

            score = 0.0
            summary = ""

            if score_match:
                score = float(score_match.group(1))
                print(f"  - [Sentiment] 분석된 점수: {score:.2f}")
            else:
                print("  - [Sentiment] 경고: 응답에서 감성 점수를 찾을 수 없습니다.")
                return 0.0, "점수 파싱 실패"

            if summary_match:
                summary = summary_match.group(1).strip()
                print(f"  - [Sentiment] 분석 근거: {summary}")
            else:
                print("  - [Sentiment] 경고: 응답에서 분석 근거를 찾을 수 없습니다.")
                summary = "근거 파싱 실패"

            return score, summary

        except Exception as e:
            print(f"  - [Sentiment] Gemini API 호출 중 오류: {e}")
            return 0.0, "API 호출 실패"

if __name__ == '__main__':
    try:
        analyzer = SentimentAnalyzer()
        
        # 비트코인에 대한 감성 분석 예시
        score_btc, summary_btc = analyzer.get_sentiment_score('BTC/KRW')
        print("\n[최종 분석 결과 - BTC/KRW]")
        print(f"  - 감성 점수: {score_btc}")
        print(f"  - 요약: {summary_btc}")

        # 이더리움에 대한 감성 분석 예시
        score_eth, summary_eth = analyzer.get_sentiment_score('ETH/KRW')
        print("\n[최종 분석 결과 - ETH/KRW]")
        print(f"  - 감성 점수: {score_eth}")
        print(f"  - 요약: {summary_eth}")

    except ValueError as e:
        print(f"오류: {e}")
