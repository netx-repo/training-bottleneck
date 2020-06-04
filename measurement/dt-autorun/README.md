# Instructions

## Notice
Make sure `git lfs` has installed and has initialized. `cifar10` dataset now is included in `distributed-training` repo
using `git lfs`.

## ImageNet
Make sure you have `imagenet` data set at `~/data`.
Here is the possible tree structure of `~/data`:
``` text
/home/ubuntu/data
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── readme.html
│   └── test_batch
├── cifar-10-python.tar.gz
└── imagenet
    ├── bounding_boxes
    ├── idxar_map.p
    ├── idxar_map_192.p
    ├── idxar_map_64.p
    ├── imagenet_2012_bounding_boxes.csv
    ├── sorted_idxar.p
    ├── train
    ├── trn_file2size.p
    ├── val_file2size.p
    └── validation
```

## Modify configurations at `training-configs` folder
Mainly adding server IPs, following file is at `training-configs/cifar10-resnet50-2p3dn/2-p3dn-resnet50-cifar10-40G.json`.
You need to change the `"nodes"` field in the config file (using EC2's private IP here). 
E.g: you have two instances: `172.31.31.15` and `172.31.29.187`, assume `172.31.29.187` is the `localhost` where we placed 
our script. Then change `nodes` to be `["localhost", "172.31.31.15"]`
``` json
{
    "comments": "unlimited bandwidth",
    "host_user": "ubuntu",
    "host_user_dir": "/home/ubuntu",
    "host_ssh_key": "~/.ssh/id_rsa",
    "docker_user_dir": "/home/cluster",
    "docker_user": "cluster",
    "docker_ssh_port": 2022,
    "docker_ssh_key": "./DockerEnv/ssh-keys/id_rsa",
    "script_path": "~/distributed-training/test_scripts/pytorch_resnet50_cifar10.py",
    "script_args": "--epochs 20",
    "nodes": ["localhost", ""],
    "nGPU": 8,
    "eth": "ens5",
    "bw_limit": "40Gbit",
    "default_bw": "100Gbit",
    "log_folder": "p3dn-ResNet50-CIFAR10"
}
```

## Run script
### single node 
``` bash 
python3 batch_run_st.py
```

### multi-nodes
``` bash
python3 docker_dt.py <config-file> 

# e.g.
python3 docker_dt.py training-configs/cifar10-resnet50-2p3dn/2-p3dn-resnet50-cifar10-40G.json
```

### mimic distributed training scripts
``` bash
python3 docker_mt.py <config-file> <debug-flag>

# e.g.
python3 docker_mt.py training-configs/mimic_config_template.json
```
* Note: logs will be saved into `chaokun_logs/<sub-dir>`, thus we need the log folder.

## Sample outputs of `docker_dt.py`
located at [example-script-output](log.example)

## Other logs
Program logs will be saved into `log_archives`