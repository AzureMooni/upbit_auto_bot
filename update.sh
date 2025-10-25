#!/bin/bash 
set -e

# 1. 스크립트가 있는 폴더로 이동
cd "$(dirname "$0")"

# 2. 환경 변수 정의
export AWS_REGION="ap-southeast-1" 
export AWS_ACCOUNT_ID="853452246030" 
export ECR_REPOSITORY="ai-commander-v2" 
export IMAGE_TAG="latest"

# 3. AWS ECR(이미지 저장소)에 로그인
echo "Logging in to Amazon ECR..." 
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# 4. 이전에 실행 중이던 컨테이너가 있다면 중지하고 삭제
echo "Stopping and removing old container..." 
docker stop upbit-bot || true 
docker rm upbit-bot || true

# 5. ECR에서 최신 버전의 Docker 이미지를 내려받기
echo "Pulling latest image from ECR..." 
docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# 6. 최신 이미지를 사용하여 새 컨테이너를 실행
# --env-file 옵션으로 서버에 있는 .env 파일의 API 키를 안전하게 주입
echo "Starting new container with .env file..." 
docker run -d \
--name upbit-bot \
--restart always \
--env-file ./.env \
$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# 7. 사용하지 않는 오래된 Docker 이미지 정리
echo "Cleaning up old images..." 
docker image prune -af

echo "Update complete. 'docker logs -f upbit-bot' "
