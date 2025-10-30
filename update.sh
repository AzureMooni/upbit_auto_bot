#!/bin/bash

# 이 스크립트는 EC2 서버에서 수동으로 실행되어야 합니다.
# 사용법: sudo bash update.sh
set -e

# 1. 환경 변수 (ECR 주소)
AWS_ACCOUNT_ID="853452246030"
AWS_REGION="ap-northeast-2"
ECR_REPOSITORY="ai-commander-v2"
IMAGE_TAG="latest"

# 2. ECR 로그인 (EC2 IAM 역할 사용)
echo "Logging in to Amazon ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# 3. 기존 컨테이너 중지 및 삭제
echo "Stopping and removing old 'upbit-bot' container..."
docker stop upbit-bot || true
docker rm upbit-bot || true

# 4. 최신 'latest' 이미지 PULL
echo "Pulling latest image: $ECR_REPOSITORY:$IMAGE_TAG..."
docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# 5. 새 컨테이너 실행
# 이 명령어는 이 스크립트와 동일한 디렉토리에 .env 파일이 있다고 가정합니다.
echo "Starting new container..."
docker run -d \
    --name upbit-bot \
    --restart always \
    --env-file ./.env \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# 6. 오래된 이미지 정리
echo "Cleaning up old docker images..."
docker image prune -af

echo "✅ Update complete."
echo "Run 'docker logs -f upbit-bot' to check the status."