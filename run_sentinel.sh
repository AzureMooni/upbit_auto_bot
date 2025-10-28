#!/bin/bash

# 스크립트의 위치를 기준으로 프로젝트 루트 디렉토리로 이동합니다.
# 이렇게 하면 crontab이 어느 위치에서 실행되든 항상 올바른 경로에서 작동합니다.
cd "$(dirname "$0")"

# 로그 디렉토리 생성
mkdir -p logs

# Sentinel 스크립트 실행
# 모든 표준 출력(stdout)과 표준 에러(stderr)를 로그 파일에 추가합니다.
echo "
--- Running Sentinel Job at $(date) ---" >> logs/sentinel.log
source venv/bin/activate
python sentinel.py >> logs/sentinel.log 2>&1
deactivate
