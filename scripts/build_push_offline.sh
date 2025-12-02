#!/usr/bin/env bash
# Build and push the offline pipeline image to ECR.
set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-sandbox-sso}"
AWS_REGION="${AWS_REGION:-ca-central-1}"
ACCOUNT_ID="${ACCOUNT_ID:-472730591037}"
IMAGE_NAME="${IMAGE_NAME:-arxplorer-offline}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"
BUILD_CONTEXT="${BUILD_CONTEXT:-offline_pipeline}"

echo "Logging in to ECR (${AWS_REGION}) with profile ${AWS_PROFILE}..."
aws ecr get-login-password --region "${AWS_REGION}" --profile "${AWS_PROFILE}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Building ${ECR_REPO}:${IMAGE_TAG} from ${BUILD_CONTEXT}..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "${BUILD_CONTEXT}"

echo "Tagging and pushing..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ECR_REPO}:${IMAGE_TAG}"
docker push "${ECR_REPO}:${IMAGE_TAG}"

echo "Done. Image: ${ECR_REPO}:${IMAGE_TAG}"
