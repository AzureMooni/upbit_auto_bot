import os
import google.generativeai as genai
from dotenv import load_dotenv

# API 키 설정
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # .env 파일에 키가 없는 경우를 대비하여, 셸 환경 변수도 확인
    api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("오류: GOOGLE_API_KEY가 설정되지 않았습니다.")
else:
    genai.configure(api_key=api_key)

    print("사용 가능한 모델 목록:")
    for m in genai.list_models():
        # generateContent 메서드를 지원하는 모델만 필터링
        if "generateContent" in m.supported_generation_methods:
            print(f"- {m.name}")
