import sys
import os
import time

import numpy as np
import torch
import torchvision
from torchvision import models


def get_size(p):
    """
    return in bytes
    """
    size = 1
    for i in p.size():
        size *= i
    return size * 4


def time_bk(model, params, idx, repeat_n):
    bk_start = [0]

    bk_times = []

    def backward_DONE(grad):
        e = torch.cuda.Event()
        e.record()
        e.synchronize()
        # print(time.time() - bk_start[0])
        bk_times.append(time.time() - bk_start[0])

    handle = params[idx][1].register_hook(backward_DONE)
    for _ in range(repeat_n):
        one_batch = torch.rand((32, 3, 224, 224)).cuda()
        outputs = model(one_batch)
        fake_target = torch.randint(1000, (32, ), dtype=torch.int64).cuda()
        loss = torch.nn.functional.cross_entropy(outputs, fake_target)

        # synchronize
        torch.cuda.synchronize()
        bk_start[0] = time.time()
        loss.backward()

    handle.remove()
    return np.mean(bk_times[5:])


def main():
    """"""
    model_map = {
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "vgg16": models.vgg16_bn
    }

    model_name = sys.argv[1]
    repeat_n = int(sys.argv[2])

    model = model_map[model_name]().cuda()

    params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append((name, p))

    bk_time_logs = []
    for i in range(len(params)-1, -1, -1):
        bk_t_at_i = time_bk(model, params, i, repeat_n)
        print(bk_t_at_i, get_size(params[i][1]))
        bk_time_logs.append((bk_t_at_i, get_size(params[i][1])))

    if not os.path.exists('./bk_time_logs'):
        os.mkdir('./bk_time_logs')

    with open("./bk_time_logs/{}.bk.log".format(model_name), 'w') as ofile:
        ofile.write("first line is for explaination; sec line: # of batches; 3rd line:"\
                    "# of layers; first col: time wait for previous layer; second col"\
                    "param size in bytes\n")
        ofile.write('1\n') # because results already averaged with multiple times
        ofile.write('{}\n'.format(len(params)))

        # write for the last layer
        # convert to microsecond
        ofile.write('{},{}\n'.format(bk_time_logs[0][0]*1e6,
                                     bk_time_logs[0][1]))

        for i in range(1, len(bk_time_logs)):
            wait_t = bk_time_logs[i][0] - bk_time_logs[i-1][0]
            wait_t *= 1e6 # convert to microsecond for consistency of the format used before
            psize = bk_time_logs[i][1]
            if wait_t < 0:
                wait_t = 0
            ofile.write('{},{}\n'.format(wait_t,
                                         psize))


if __name__ == '__main__':
    main()
