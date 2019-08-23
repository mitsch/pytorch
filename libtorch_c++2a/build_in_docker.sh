#!/bin/bash

set -e

PROJECT="libtorch20"

# Location of this script relative to the caller
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR_INTERNAL="/root/$(basename "${DIR}")"
WORKSPACE="/root/$(basename "${DIR}")"_build

# Hash the absolute path to use it as identifier
ID="$( echo "${DIR}" | md5sum | cut -f1 -d' ' )"

# Commit ID from git
GIT_COMMIT="$(git -C "${DIR}" log --pretty=format:'%H' -n 1)"

IMAGE_TAG="${PROJECT}:build-stack-commit${GIT_COMMIT}"

# build build-stack stage via docker
nvidia-docker build --network host --tag ${IMAGE_TAG} --target build-stack -f $DIR/Dockerfile $DIR

CONTAINER_NAME="${PROJECT}-${USER}-build-env-${ID}"

# test if some container with the name CONTAINER_NAME is already running
if [ -z "$(nvidia-docker ps -f "name=${CONTAINER_NAME}" | tail -n+2)" ]; then
    echo "Container is not running ..." >> /dev/stderr
    # test if some container with the name CONTAINER_NAME exists
    if [ -z "$(nvidia-docker ps -a -f "name=${CONTAINER_NAME}" | tail -n+2)" ]; then
        echo "Container does not exist ..." >> /dev/stderr
        nvidia-docker run -t -d --name ${CONTAINER_NAME} --net host -v "${DIR}:${DIR_INTERNAL}:ro" ${IMAGE_TAG} bash
    fi
    nvidia-docker start ${CONTAINER_NAME}
fi

nvidia-docker exec -t -w /root ${CONTAINER_NAME} ldconfig
nvidia-docker exec -t -w /root ${CONTAINER_NAME} mkdir -p "${WORKSPACE}"
nvidia-docker exec -t -w ${WORKSPACE} ${CONTAINER_NAME} cmake "${DIR_INTERNAL}"
nvidia-docker exec -t -w ${WORKSPACE} ${CONTAINER_NAME} cmake --build .
