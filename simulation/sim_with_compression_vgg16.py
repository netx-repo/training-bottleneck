import os

from sim_with_compression_comm import get_bk_sim_time, VecAddCost


def main():
    sd_batch_time = {
        "resnet101": 0.19200599479476405,
        "resnet50": 0.12808580219559176,
        "vgg16": 0.19452352295080985
    }

    vecAddEst = VecAddCost('./cuda_vec_add.log')
    # experiments = experiments[:2]
    bk_logfile = "./bk_time_logs/vgg16.bk.log"
    model_name = 'vgg16'
    batch_t = sd_batch_time[model_name]

    for comp in [1, 2, 5, 10, 50, 100]:
    # for comp in [100]:
        print('*'*10, 'compression', comp)
        for N in [16, 32, 64]:
            for linkspeed in [100e9, 40e9, 25e9, 10e9, 1e9]:
                slowdown_f = 10

                coll_bk_time, coll_overhead = get_bk_sim_time(bk_logfile,
                                                              N, linkspeed, vecAddEst,
                                                              comp, slowdown_f)

                print('\t', model_name,
                      N, linkspeed/1e9, 'Gbps, scaling factor',
                      batch_t / (batch_t + coll_overhead))

        print('*'*10, 'compression', comp)


if __name__ == '__main__':
    main()
