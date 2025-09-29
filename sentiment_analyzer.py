import os
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio

class SentimentAnalyzer:
    def __init__(self):
        load_dotenv(os.path.join(os.path.dirname(__file__), 'config', '.env'))
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')

        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set in config/.env")

        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    async def analyze_market_sentiment(self, ticker: str):
        prompt = f"현재 '{ticker}' 코인 및 전체 암호화폐 시장에 대한 최신 뉴스, 소셜 미디어 동향, 기술적 분석가들의 의견을 종합적으로 분석해줘. 그리고 현재 시장의 투자 심리를 '매우 긍정적', '긍정적', '중립', '부정적', '매우 부정적' 중 하나로 평가하고, 그 이유를 한 문장으로 요약해줘."
        
        try:
            response = await self.model.generate_content_async(prompt)
            sentiment_text = response.text.strip()
            
            # Gemini의 답변에서 최종 '평가'와 '이유'를 추출
            # 예시 응답: "긍정적. 비트코인 현물 ETF 승인 기대감으로 시장 전반에 매수 심리가 강합니다."
            
            sentiment_mapping = {
                "매우 긍정적": "매우 긍정적",
                "긍정적": "긍정적",
                "중립": "중립",
                "부정적": "부정적",
                "매우 부정적": "매우 부정적"
            }
            
            sentiment = "중립"
            reason = "분석 결과가 명확하지 않습니다."

            for key, value in sentiment_mapping.items():
                if key in sentiment_text:
                    sentiment = value
                    # 평가 이후의 텍스트를 이유로 간주
                    reason_start_index = sentiment_text.find(key) + len(key)
                    reason = sentiment_text[reason_start_index:].strip()
                    if reason.startswith('.'): # Remove leading dot if present
                        reason = reason[1:].strip()
                    if not reason: # If no specific reason found after sentiment, use a default
                        reason = f"{key}으로 평가됩니다."
                    break
            
            return sentiment, reason
        except Exception as e:
            print(f"Error analyzing sentiment for {ticker}: {e}")
            return "중립", f"감성 분석 중 오류 발생: {e}"

if __name__ == '__main__':
    # .env 파일 생성 (테스트용)
    env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("GEMINI_API_KEY=YOUR_GEMINI_API_KEY")
        print(f"Created a dummy .env file at {env_path}. Please replace YOUR_GEMINI_API_KEY with your actual Gemini API key.")
    
    async def test_sentiment_analyzer():
        try:
            analyzer = SentimentAnalyzer()
            sentiment, reason = await analyzer.analyze_market_sentiment("BTC/KRW")
            print(f"BTC/KRW Sentiment: {sentiment}, Reason: {reason}")
        except ValueError as e:
            print(f"Configuration Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during sentiment analysis test: {e}")

    asyncio.run(test_sentiment_analyzer())