#!/bin/bash

CONTAINER_REGISTRY="zarzen/horovod-mod:1.0"

rm -rf ssh-keys

id=$(sudo docker create $CONTAINER_REGISTRY) && \
sudo docker cp ${id}:/home/cluster/.ssh -> ./ssh-keys.tar && \
sudo docker rm -v ${id}

tar -xvf ssh-keys.tar
mv ./.ssh ssh-keys
rm ssh-keys.tar