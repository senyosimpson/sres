#!/usr/bin/env bash
gradient jobs create \
--name "$1" \
--container ${CONTAINER_NAME} \
--registryUsername ${DOCKERHUB_USERNAME} \
--registryPassword ${DOCKERHUB_PASSWORD} \
--machineType "P5000" \
--command "./baseline.sh" \
--ignoreFiles trained