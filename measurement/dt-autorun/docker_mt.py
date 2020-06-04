""" running inside docker: mimic training process
"""
import json
import os
import subprocess
from os.path import expanduser, join
import sys

from pssh.clients.native import ParallelSSHClient
from pssh.clients.native.single import SSHClient


class MimicExp:
    def __init__(self, config, debug=0):
        self.base_logdir = "autorun/chaokun_logs"
        self._parse_config(config)
        self.debug = debug

    def _parse_config(self, config):
        self.host_user_dir = config["host_user_dir"]
        self.docker_user_dir = config["docker_user_dir"]
        self.docker_user = config["docker_user"]
        self.docker_ssh_port = config["docker_ssh_port"]
        self.nodes = config['nodes']
        self.nGPU = config['nGPU']  # for each machine
        self.eth = config['eth']  # name if NIC
        self.host_key = config["host_ssh_key"]
        self.docker_key = config["docker_ssh_key"]

    def _init_host_ssh(self):
        """connect to all nodes"""
        self.pClinet = ParallelSSHClient(self.nodes, pkey=self.host_key)

    def _init_docker_ssh(self):
        self.docker0 = SSHClient("localhost", user=self.docker_user,
                                 port=self.docker_ssh_port, pkey=self.docker_key)

    def _ini_host_env(self):
        """download log folder from aws-s3"""
        sentinel_cmd = "mkdir ~/autorun; "
        self._p_exe(sentinel_cmd)

        check_logs_cmd = "cd ~/autorun; mkdir tmp/; cd tmp/; rm mimic_env_setup.sh; "\
            "wget https://gist.githubusercontent.com/zarzen/012c2aa2a1c833e5bf1aeb379bbb9e93/raw/71dec6db3138dcc7e5318598bc770c6ce296b9a4/mimic_env_setup.sh; "\
            "/bin/bash mimic_env_setup.sh"
        self._p_exe(check_logs_cmd)

    def _p_exe(self, cmd):
        output = self.pClinet.run_command(cmd)
        for host, host_output in output.items():
            for line in host_output.stdout:
                print("Host [%s] - %s" % (host, line))
            for line in host_output.stderr:
                print("Host [%s] - %s" % (host, line))

    def _docker_exe(self, cmd):
        _channel, _host, _stdout, _stderr, _ = self.docker0.run_command(cmd)
        for line in _stdout:
            print("[{}] {}".format(_host, line))
        for line in _stderr:
            print("[{}] err {}".format(_host, line))

    def _start_containers(self):
        stop_cmd = "docker kill $(docker ps -q)"
        pull_cmd = "docker pull zarzen/horovod-mod:1.0"
        start_cmd = "sudo docker run --gpus all --network=host --detach --ipc=host "\
            "-v {}/autorun/chaokun_logs:{}/chaokun_logs "\
            "zarzen/horovod-mod:1.0".format(self.host_user_dir, self.docker_user_dir
                                            )
        self._p_exe(stop_cmd)
        self._p_exe(pull_cmd)
        self._p_exe(start_cmd)

    def run(self):
        self._init_host_ssh()
        self._ini_host_env()
        self._start_containers()
        self._init_docker_ssh()

        exp_folders = os.listdir(join(self.host_user_dir, self.base_logdir))
        # first 10 folder for debugging for now
        if self.debug:
            print("debug mode, experimentally run 3 configurations")
            exp_folders = exp_folders[:3]
        else:
            print("no debug flag, try to experiments on all configurations")

        for idx, _folder in enumerate(exp_folders):
            self._run_once(_folder)
            print("*"*10,
                  "Completed {}/{}".format(idx+1, len(exp_folders)),
                  "*"*10)

    def _run_once(self, folder_name):
        """ run with 
        """
        # read the orginal experiment config to get the bw limit
        folder_path = join(self.host_user_dir,
                           self.base_logdir, folder_name)
        if not os.path.isdir(folder_path) or \
            not os.path.exists(join(folder_path, "config.json")):
            return

        with open(join(folder_path, "config.json")) as ifile:
            config = json.load(ifile)
            bw_limit = config['bw_limit']
            print("mimic training with folder {} at bw {}".format(
                folder_name, bw_limit))
            self._bw_ctl(bw_limit)
            cpu_p, net_p = self._exe_res_monitor(folder_path)
            print(">"*10, 'launched CPU & Network monitoring')
            mt_cmd = self._build_mpirun_cmd(config, folder_name)
            print("executing mimic training command:\n", mt_cmd)

            # import time
            # time.sleep(2)
            self._docker_exe(mt_cmd)
            cpu_p.terminate()
            net_p.terminate()

    def _build_mpirun_cmd(self, config, folder_name):
        """"""
        folder_path = join(self.docker_user_dir,
                           "chaokun_logs", folder_name)
        nNodes = len(config['nodes'])
        nGPU = config['nGPU']
        if self.debug:
            # because test env only has two nodes with 1 GPU on each
            nNodes = 2
            nGPU = 1
        IPs = self.nodes[:nNodes]
        hostsStr = ",".join(["{}:{}".format(ip, nGPU) for ip in IPs])
        cmd = [
            "mpirun", "-np", str(nNodes*nGPU),
            "-H", hostsStr,
            "-bind-to", "none",
            "-map-by", "slot",
            "-x", "LD_LIBRARY_PATH=/usr/local/cuda/lib64",
            "-x", "NCCL_DEBUG=INFO",
            "-x", "NCCL_SOCKET_IFNAME=^lo,docker,ens4",
            "-mca", "btl_tcp_if_exclude lo,docker,ens4",
            self.docker_user_dir +
            "/mimic_dt/build/mdt_allreduce_perf",
            "-b 500M -e 500M -f 2 -g 1 -c 0 -w 0",
            "-l", join(folder_path, "log_for_dt_mimic.txt"),
            "|& grep -v \"Read -1\""
        ]
        return " ".join(cmd)

    def _bw_ctl(self, bw_limit):
        del_cmd = "sudo tc qdisc del dev {} root tbf rate 40Gbit latency 400ms burst 3000kbit".format(
            self.eth)
        # if self.bw_limit = "" then we don't execute the add_cmd
        add_cmd = "sudo tc qdisc add dev {} root tbf rate {} latency 400ms burst 3000kbit".format(
            self.eth, bw_limit)
        print('deleting old bw limit')
        self._p_exe(del_cmd)
        print('confirm the bw limit deleted (should see error when redoing del)')
        self._p_exe(del_cmd)
        if bw_limit != "":
            self._p_exe(add_cmd)

    def _exe_res_monitor(self, tg_folder):
        """ execute cpu and network bandwidth monitor
        """
        # record existing logs
        cpu_monitor_script = expanduser("~/autorun/monitor_cpu.py")
        net_monitor_script = expanduser("~/autorun/monitor_net.py")
        cpu_p = subprocess.Popen(["python3", cpu_monitor_script,
                                  join(tg_folder, "mt_cpu.log")],
                                 stdout=subprocess.DEVNULL)
        net_p = subprocess.Popen(["python3", net_monitor_script,
                                  join(tg_folder, "mt_net.log")],
                                 stdout=subprocess.DEVNULL)
        return cpu_p, net_p

    def __del__(self):
        stop_cmd = "docker kill $(docker ps -q)"
        self._p_exe(stop_cmd)


def main():
    if len(sys.argv) < 2:
        print("please specify a configuration file, which contains server IPs etc")
    conf_path = sys.argv[1]
    debug_flag = 1 if len(sys.argv) > 2 else 0

    with open(conf_path) as ifile:
        config = json.load(ifile)
    exp = MimicExp(config, debug=debug_flag)
    exp.run()


if __name__ == "__main__":
    main()
