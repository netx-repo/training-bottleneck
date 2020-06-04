export USER="cluster"
export CONTAINER_NAME="horovod-mod"
export CONTAINER_REGISTRY="zarzen/horovod-mod:1.0"

image: Dockerfile
	sudo docker build --tag $(CONTAINER_REGISTRY) .

push-image: image
	sudo docker push $(CONTAINER_REGISTRY)

copy-keys:
	./copy_ssh_keys.sh