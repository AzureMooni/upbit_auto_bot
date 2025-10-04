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

    def get_fear_greed_index(self, ticker: str) -> tuple[int, str]:
        """
        주어진 티커에 대한 시장의 공포-탐욕 지수를 0-100 사이의 점수로 반환합니다.

        Args:
            ticker (str): 분석할 암호화폐 티커 (예: 'BTC/KRW')

        Returns:
            tuple[int, str]: (공포-탐욕 지수, 시장 핵심 내러티브). 오류 시 (50, "분석 실패")
        """
        print(f"  - [Sentiment] '{ticker}' 시장의 공포-탐욕 지수 분석 중...")
        try:
            prompt = f"""
            당신은 최고의 암호화폐 퀀트 분석가입니다.
            현재 '{ticker}'를 포함한 전반적인 암호화폐 시장의 투자 심리를 분석해주세요.
            최신 뉴스, 기술적 지표(RSI, 변동성 등), 온체인 데이터, 소셜 미디어 동향을 종합적으로 고려해야 합니다.
            
            분석 결과를 바탕으로, 현재 시장의 공포-탐욕 지수를 0(극단적 공포)에서 100(극단적 탐욕) 사이의 정수 점수로 평가하고, 현재 시장을 지배하는 핵심 내러티브(키워드)가 무엇인지 1~2문장으로 요약해주세요.
            
            출력 형식은 반드시 아래와 같이 "지수: [숫자]" 와 "핵심 내러티브: [요약]" 형식으로 작성해주세요.
            
            지수: [여기에 0에서 100 사이의 숫자만 삽입]
            핵심 내러티브: [여기에 핵심 내러티브 요약 삽입]
            """
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # 점수와 근거를 파싱
            score_match = re.search(r"지수:\s*(\d+)", response_text)
            narrative_match = re.search(r"핵심 내러티브:\s*(.*)", response_text, re.DOTALL)

            score = 50  # 기본값: 중립
            narrative = ""

            if score_match:
                score = int(score_match.group(1))
                print(f"  - [Sentiment] 분석된 공포-탐욕 지수: {score}")
            else:
                print("  - [Sentiment] 경고: 응답에서 공포-탐욕 지수를 찾을 수 없습니다.")
                return 50, "지수 파싱 실패"

            if narrative_match:
                narrative = narrative_match.group(1).strip()
                print(f"  - [Sentiment] 시장 핵심 내러티브: {narrative}")
            else:
                print("  - [Sentiment] 경고: 응답에서 핵심 내러티브를 찾을 수 없습니다.")
                narrative = "내러티브 파싱 실패"

            return score, narrative

        except Exception as e:
            print(f"  - [Sentiment] Gemini API 호출 중 오류: {e}")
            return 50, "API 호출 실패"

if __name__ == '__main__':
    try:
        analyzer = SentimentAnalyzer()
        
        score, narrative = analyzer.get_fear_greed_index('BTC/KRW')
        print("\n[최종 분석 결과 - BTC/KRW]")
        print(f"  - 공포-탐욕 지수: {score}")
        print(f"  - 핵심 내러티브: {narrative}")

    except ValueError as e:
        print(f"오류: {e}")
