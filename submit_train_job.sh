gradient jobs create \
--name "$0" \
--container "docker.io/senyo/sres:latest" \
--registryUsername ${DOCKERHUB_USRENAME} \
--registryPassword ${DOCKERHUB_PASSWORD} \
--machineType "P5000" \
--command "sh train.sh"
