export IMAGE_NAME=${CI_PROJECT_NAME}
docker-compose -f ${CI_PROJECT_NAME}.yml -p ${CI_PROJECT_NAME} down --remove-orphans
docker rmi -f ${CI_PROJECT_NAME}
gunzip -c ${CI_PROJECT_NAME}.tgz | docker load
docker-compose -f ${CI_PROJECT_NAME}.yml -p ${CI_PROJECT_NAME} up -d --remove-orphans