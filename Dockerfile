# 1. Base Image
FROM python:3.11-slim

# 2. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Working Directory
WORKDIR /app

# 4. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pyjwt==2.3.0
RUN pip install pyupbit
RUN pip install --no-cache-dir tensorflow

# [수정] 애플리케이션 코드를 먼저 복사
COPY . /app

# [추가] AI 모델 및 필수 데이터 파일을 이미지에 복사합니다.
# 'models' 디렉토리와 'data' 디렉토리가 있다고 가정합니다.
# 만약 파일/폴더 이름이 다르다면 이 부분을 수정해야 합니다.
COPY models/ /app/models/
COPY data/ /app/data/
COPY specialist_stats.json /app/specialist_stats.json

# 7. Final Entrypoint
ENTRYPOINT ["python", "live_trader.py"]