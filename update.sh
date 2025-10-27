#!/bin/bash

# This script automates the process of updating the Docker container on the EC2 server.

# --- Configuration ---
# IMPORTANT: Fill in your AWS Account ID below.
AWS_ACCOUNT_ID="YOUR_AWS_ACCOUNT_ID"
AWS_REGION="ap-northeast-2"
ECR_REPOSITORY_NAME="upbit-auto-bot"
# Use the commit hash as the image tag for precise versioning
IMAGE_TAG=${1:-latest} # Default to 'latest' if no argument is provided
CONTAINER_NAME="upbit-bot-container"

# Full Image URI in AWS ECR
ECR_IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}"

# --- Execution ---

set -e # Exit immediately if a command exits with a non-zero status.

echo "🚀 Starting deployment script for tag: ${IMAGE_TAG}..."

# 1. Log in to AWS ECR
echo "1/5: Logging in to AWS ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
echo "✅ ECR login successful."

# 2. Pull the latest image from ECR
echo "\n2/5: Pulling image from ECR: ${ECR_IMAGE_URI}"
docker pull ${ECR_IMAGE_URI}
echo "✅ Image pull successful."

# 3. Stop the existing container if it is running
if [ $(docker ps -q -f name=${CONTAINER_NAME}) ]; then
    echo "\n3/5: Stopping existing container..."
    docker stop ${CONTAINER_NAME}
    echo "✅ Container stopped."
else
    echo "\n3/5: No running container found with the name ${CONTAINER_NAME}. Skipping stop."
fi

# 4. Remove the stopped container
if [ $(docker ps -a -q -f name=${CONTAINER_NAME}) ]; then
    echo "\n4/5: Removing existing container..."
    docker rm ${CONTAINER_NAME}
    echo "✅ Container removed."
else
    echo "\n4/5: No container found with the name ${CONTAINER_NAME}. Skipping removal."
fi

# 5. Run the new container from the updated image
echo "\n5/5: Running the new container..."
docker run -d \
    --name ${CONTAINER_NAME} \
    --env-file ./.env \
    --restart always \
    ${ECR_IMAGE_URI}

echo "\n🎉 Deployment complete! The new bot container is now running with tag [${IMAGE_TAG}]."
