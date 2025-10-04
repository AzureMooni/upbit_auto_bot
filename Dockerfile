# Stage 1: Builder
# 이 단계에서는 빌드에 필요한 의존성을 설치하고 패키지를 빌드합니다.
FROM python:3.12-slim as builder

WORKDIR /app

# 빌드에 필요한 패키지(git 포함) 설치 및 캐시 정리
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

# requirements.txt를 먼저 복사하여 Docker의 레이어 캐싱을 활용
COPY requirements.txt .

# 가상 환경 생성 및 활성화, 그리고 의존성 설치
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# --- 

# Stage 2: Final Image
# 이 단계에서는 빌드된 결과물만 가져와 최종 이미지를 만듭니다.
FROM python:3.12-slim

WORKDIR /app

# 보안을 위해 권한 없는 사용자 생성 및 전환
RUN useradd --create-home appuser
USER appuser

# Builder 단계에서 설치된 패키지만 복사
COPY --from=builder /opt/venv /opt/venv

# 애플리케이션 코드 복사
COPY . .

# 가상 환경의 파이썬을 사용하도록 PATH 설정
ENV PATH="/opt/venv/bin:$PATH"

# 기본 CMD는 도움말을 표시하도록 설정 (실제 실행은 docker run에서 오버라이드)
CMD ["python", "main.py", "--help"]
