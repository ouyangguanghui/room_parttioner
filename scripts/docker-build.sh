#!/usr/bin/env bash
# Docker 镜像构建 / 推送脚本
# 用法:
#   ./scripts/docker-build.sh build          # 构建 HTTP 镜像
#   ./scripts/docker-build.sh build-lambda   # 构建 Lambda 镜像
#   ./scripts/docker-build.sh push           # 推送 HTTP 镜像到 ECR
#   ./scripts/docker-build.sh push-lambda    # 推送 Lambda 镜像到 ECR
#   ./scripts/docker-build.sh all            # 构建并推送全部

set -euo pipefail

# ---- 配置 ----
IMAGE_NAME="${IMAGE_NAME:-room-partitioner}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
AWS_REGION="${AWS_REGION:-cn-north-1}"
ECR_REGISTRY="${ECR_REGISTRY:-}"  # 例如: 123456789012.dkr.ecr.cn-north-1.amazonaws.com.cn

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---- 函数 ----

build_http() {
    echo "==> Building HTTP image: ${IMAGE_NAME}:${IMAGE_TAG}"
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "${PROJECT_DIR}"
    echo "==> Done: ${IMAGE_NAME}:${IMAGE_TAG}"
}

build_lambda() {
    echo "==> Building Lambda image: ${IMAGE_NAME}-lambda:${IMAGE_TAG}"
    docker build -f "${PROJECT_DIR}/Dockerfile.lambda" \
        -t "${IMAGE_NAME}-lambda:${IMAGE_TAG}" "${PROJECT_DIR}"
    echo "==> Done: ${IMAGE_NAME}-lambda:${IMAGE_TAG}"
}

ecr_login() {
    if [ -z "${ECR_REGISTRY}" ]; then
        echo "ERROR: ECR_REGISTRY not set. Export it or set in script."
        exit 1
    fi
    echo "==> Logging into ECR: ${ECR_REGISTRY}"
    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${ECR_REGISTRY}"
}

push_http() {
    ecr_login
    local remote="${ECR_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "==> Pushing HTTP image: ${remote}"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${remote}"
    docker push "${remote}"
    echo "==> Done: ${remote}"
}

push_lambda() {
    ecr_login
    local remote="${ECR_REGISTRY}/${IMAGE_NAME}-lambda:${IMAGE_TAG}"
    echo "==> Pushing Lambda image: ${remote}"
    docker tag "${IMAGE_NAME}-lambda:${IMAGE_TAG}" "${remote}"
    docker push "${remote}"
    echo "==> Done: ${remote}"
}

# ---- 入口 ----

case "${1:-help}" in
    build)        build_http ;;
    build-lambda) build_lambda ;;
    push)         push_http ;;
    push-lambda)  push_lambda ;;
    all)
        build_http
        build_lambda
        push_http
        push_lambda
        ;;
    *)
        echo "Usage: $0 {build|build-lambda|push|push-lambda|all}"
        exit 1
        ;;
esac
