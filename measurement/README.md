# Instructions
Do record timestamps of each operation in distributed training requires modified version of horovod located at `horovod-modified-timing` directory. 

We also provide a `docker image` with compiled horovod and other neccessary packages. You can obtain the image using `docker pull zarzen/horovod-mod:1.0` for customized distributed training. 

To repeat our measure study, automatic running scripts and training configurations are included `dt-autorun` folder. Please refer to detailed instructions inside `dt-autorun` for running on AWS.

