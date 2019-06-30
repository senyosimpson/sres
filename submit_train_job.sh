gradient jobs create \
--name "$0" \
--container ${CONTAINER_NAME} \
--registryUsername ${DOCKERHUB_USERNAME} \
--registryPassword ${DOCKERHUB_PASSWORD} \
--machineType "P5000" \
--command "sh train.sh"
