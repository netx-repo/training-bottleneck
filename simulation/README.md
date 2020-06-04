# Instructions
The first step is to generate the backward timing logs for second step, simulation. The backward timing logs contain relevative timestamp of layerwise backward process. So the simulation process can know when to start allreduce process.


## 1. generate backward logs
Note: 
1. You can also use logs in `bk_time_log`, instead of generating your own.
2. only `resnet50`, `resnet101` and `vgg16` are supported now.

```
python layerwise_bk_done.py <model-name> <repeat-time>

e.g.
python layerwise_bk_done.py resnet50 10
```

## 2. simulation 
* `simulation/sim_with_compression_resnet50.py` -> for `resnet50`
* `simulation/sim_with_compression_resnet101.py` -> for `resnet101`
* `simulation/sim_with_compression_vgg16.py` -> for `vgg16`

