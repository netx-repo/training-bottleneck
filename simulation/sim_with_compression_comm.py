import multiprocessing as mp
import numpy as np
import time
import os
import json
import subprocess

import pprint

pp = pprint.PrettyPrinter(indent=2)

class BackwardProc(mp.Process):
    def __init__(self, events, coll_pipe, slowdown=1e3):
        """
        :param events: denote a list of pairs, each pair (wait_time, layer_size)
                        wait_time mimic the backward computation time, size in bytes
        :param coll_queue: send the collective operation request to
        """
        super(BackwardProc, self).__init__()
        self.events = events
        self.coll_pipe = coll_pipe
        self.slowdown = slowdown

    def run(self):
        # heuristic batching
        timeout = 0.005 # 5ms
        buff_size = 64 * 1024 * 1024 # 64MB

        acc_time = 0
        acc_buff = 0
        for e in self.events:
            wait_t = e[0] * self.slowdown # enlarge time period

            if e[0] > 0:
                t = time.time()
                if wait_t > 0.00003:
                    # more accurate in ms, only has 0.03 ms error
                    while time.time() - t < wait_t:
                        pass

            acc_time += e[0]
            acc_buff += e[1]

            if acc_time >= timeout or acc_buff >= buff_size:
                self.coll_pipe.send(acc_buff)
                acc_time = 0
                acc_buff = 0

        if acc_buff > 0:
            self.coll_pipe.send(acc_buff)
            acc_time, acc_buff = 0, 0

        # DONE signal
        self.coll_pipe.send(-1)



class AllReduceProc(mp.Process):
    def __init__(self, coll_pipe, N, bw, vec_add_est, grad_compression, slowdown=1e3):
        super(AllReduceProc, self).__init__()

        self.coll_pipe = coll_pipe
        self.N = N
        self.bw = bw
        self.vec_add_est = vec_add_est
        self.slowdown = slowdown
        self.grad_compression = grad_compression

    def run(self):
        # sender_pipe, add_pipe = mp.Pipe()
        # add_proc = VecAddProc(add_pipe, self.vec_add_est, self.slowdown)
        # add_proc.start()

        total_allreduce_size = 0
        total_net_time = 0
        while True:
            # size in bytes
            size = self.coll_pipe.recv()
            total_allreduce_size += size
            if size < 0:
                return
            else:
                # first stage reduce-scatter
                net_cost = (size * 8 * (self.N-1) / self.N) / self.bw
                net_cost /= self.grad_compression
                add_cost = (self.N - 1) * self.vec_add_est.est(size / self.N)
                first_stage_cost = net_cost + add_cost
                first_stage_cost *= self.slowdown
                if first_stage_cost > 0.00003:
                    t = time.time()
                    while time.time() - t < first_stage_cost:
                        pass

                # second stage allgather
                net_cost = (size * 8 * (self.N - 1) / self.N) / self.bw
                net_cost /= self.grad_compression
                total_net_time += net_cost
                net_cost *= self.slowdown

                if net_cost > 0.00003:
                    t = time.time()
                    while time.time() - t < net_cost:
                        pass



class VecAddCost:
    def __init__(self, logfile):
        xp, fp = [], []
        with open(logfile) as ifile:
            for line in ifile:
                x, y = line.strip('\n').split(',')
                xp.append(x)
                fp.append(y)
        self.xp = np.array(xp).astype(np.float32)
        self.fp = np.array(fp).astype(np.float32)

    def est(self, size):
        """
        :param size: size in bytes
        """
        est_cost = np.interp(size, self.xp, self.fp)
        return est_cost


def read_backward_log(logfile):

    with open(logfile) as ifile:
        _ = ifile.readline() # comment
        batch_n = int(ifile.readline().strip('\n'))
        layer_n = int(ifile.readline().strip('\n'))

        data = []

        for i in range(batch_n):
            bk_events = []
            for l in range(layer_n):
                line = ifile.readline()
                wt, s = line.strip('\n').split(',')
                wt, s = float(wt), int(s)
                bk_events.append((wt/1e6, s))

            data.append(bk_events)
        return data


def get_bk_sim_time(logfile_path, N, linkspeed, add_cost, compression, slowdown_f=10):

    # those backward events already averaged
    bk_events = read_backward_log(logfile_path)

#     one_batch_times = []
    coll_bktimes = []
    overheads = []
    for es in bk_events:

        # compute the backward and gradient sync time
        # sum to get backward total time
        bk_t = np.sum(np.array(es)[:, 0])

        bk_pipe, coll_pipe = mp.Pipe()

        bk_p = BackwardProc(es, bk_pipe, slowdown_f)
        coll_p = AllReduceProc(coll_pipe, N, linkspeed, add_cost, compression, slowdown_f)

        t1 = time.time()
        bk_p.start()
        coll_p.start()

        bk_p.join()
        coll_p.join()
        bk_coll_time = ((time.time() - t1) / slowdown_f)
#         print('bk coll time', bk_coll_time)
        coll_bktimes.append(bk_coll_time)

        overheads.append(bk_coll_time - bk_t)
        if bk_coll_time - bk_t < 0 :
            print('not possible')

    return np.mean(coll_bktimes), np.mean(overheads)
