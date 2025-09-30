import os
import google.generativeai as genai
from dotenv import load_dotenv

class SentimentAnalyzer:
    """
    Gemini API를 사용하여 특정 암호화폐에 대한 시장 감성을 분석합니다.
    """
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-pro-latest')

    def analyze(self, ticker: str) -> str:
        """
        주어진 티커에 대한 시장 감성을 분석하고 결과를 반환합니다.

        Args:
            ticker (str): 분석할 암호화폐 티커 (예: 'BTC/KRW')

        Returns:
            str: 분석 결과 ('Positive', 'Negative', 'Neutral')
        """
        print(f"Gemini 정보 분석가: '{ticker}'에 대한 시장 감성 분석을 시작합니다...")
        try:
            prompt = f"""
            당신은 암호화폐 시장 분석 전문가입니다. 
            현재 '{ticker}' 암호화폐에 대한 최신 뉴스, 소셜 미디어 동향, 커뮤니티 여론을 종합적으로 분석해주세요. 
            분석 결과를 바탕으로 현재 시장의 전반적인 감성(Sentiment)을 'Positive', 'Negative', 'Neutral' 중 하나로만 평가해주세요. 
            다른 설명은 필요 없습니다. 오직 세 단어 중 하나로만 답해주세요.
            """
            response = self.model.generate_content(prompt)
            
            # 응답 텍스트에서 키워드 추출
            result_text = response.text.strip()
            if 'Positive' in result_text:
                print("분석 결과: 긍정적 (Positive)")
                return 'Positive'
            elif 'Negative' in result_text:
                print("분석 결과: 부정적 (Negative)")
                return 'Negative'
            else:
                print("분석 결과: 중립 (Neutral)")
                return 'Neutral'

        except Exception as e:
            print(f"Gemini API 호출 중 오류가 발생했습니다: {e}")
            return 'Neutral' # 오류 발생 시 중립으로 간주

if __name__ == '__main__':
    # GOOGLE_API_KEY 환경변수 설정 필요
    # 예: export GOOGLE_API_KEY='YOUR_API_KEY'
    analyzer = SentimentAnalyzer()
    
    # 비트코인에 대한 감성 분석 예시
    sentiment = analyzer.analyze('BTC/KRW')
    print(f"\n최종 분석된 '/BTC/KRW'의 시장 감성: {sentiment}")

    # 이더리움에 대한 감성 분석 예시
    sentiment_eth = analyzer.analyze('ETH/KRW')
    print(f"\n최종 분석된 'ETH/KRW'의 시장 감성: {sentiment_eth}")
