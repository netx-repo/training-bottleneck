#!/bin/bash

sudo docker run --gpus all --network=host --detach \
    -v /home/ubuntu/autorun/distributed-training:/home/cluster/distributed-training \
    zarzen/horovod-mod:1.0
