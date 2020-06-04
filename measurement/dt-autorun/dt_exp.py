import os
from os.path import expanduser, join, abspath
import subprocess
import datetime
import shutil
import paramiko


class ExpRunner:

    def __init__(self, python_exe: str,
                    script_path: str, script_args: str,
                    nodes: list, nGPU: str, eth: str, bw_limit: str,
                    log_folder=None) -> None:
        """"""
        self.python_bin = abspath(expanduser(python_exe))
        self.script_path = abspath(expanduser(script_path))
        self.script_args = script_args
        self.nodes = nodes
        self.nGPU = nGPU # for each machine
        self.eth = eth # name if NIC
        self.bw_limit = bw_limit
        self.log_folder = log_folder
        self.key = paramiko.RSAKey.from_private_key_file(expanduser("~/.ssh/id_rsa"))
        self._init_ssh()
        self.exist_logs = self._get_logs()
    
    def _init_ssh(self):
        print('='*10, 'initializing ssh connections')
        self.clients = []
        for node in self.nodes:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=node, username="ubuntu", pkey=self.key)
            self.clients.append((node, client))
            print('IP', node, 'DONE')
        print('='*10, 'initialization for ssh clients DONE')

    def _init_host_env(self):
        """"""
        for ip, cli in self.clients:
            check_cmd = "cd ~/; ls|grep distributed-training"
            _, stdout, stderr = cli.exec_command(check_cmd)
            if stdout.read() != b"":
                git_pull = "cd ~/distributed-training; git pull"
                self._exec_cli_cmd(cli, git_pull, '{}: git pull'.format(ip))
            else:
                cmd = "cd ~/; "\
                    "git clone https://github.com/zarzen/distributed-training.git"
                self._exec_cli_cmd(cli, cmd, "{}: clone training scripts".format(ip))

    def _exec_cli_cmd(self, cli, cmd, msg=None):
        if msg:
            print('>'*10, msg, '<'*10)
        _, stdout, stderr = cli.exec_command(cmd)
        print('cmd stdout: ', stdout.read().decode('utf-8'),
              "cmd stderr: ", stderr.read().decode('utf-8'))
        if msg:
            print('>'*10, 'DONE', msg, '<'*10)
    
    def bandwith_control(self):
        """
        """
        del_cmd = "sudo tc qdisc del dev {} root tbf rate 40Gbit latency 400ms burst 3000kbit".format(self.eth)
        # if self.bw_limit = "" then we don't execute the add_cmd
        add_cmd = "sudo tc qdisc add dev {} root tbf rate {} latency 400ms burst 3000kbit".format(self.eth, self.bw_limit)
        for (ip, cli) in self.clients:
            # try to delete rate limit
            stdin, stdout, stderr = cli.exec_command(del_cmd)
            print(ip, ":", stdout.read(), stderr.read())
            stdin, stdout, stderr = cli.exec_command(del_cmd)
            print(ip, ":", stdout.read(), stderr.read())

            if self.bw_limit:
                print(ip, ": adding bandwidth limit", add_cmd)
                stdin, stdout, stderr = cli.exec_command(add_cmd)
                print(ip, ':', stdout.read(), stderr.read())

    def exe_dist_train(self) -> subprocess.Popen:
        """ execute distributed training script at rank0
        :return process:
        """
        train_cmd = self.build_train_cmd()
        print("Exec:", " ".join(train_cmd))
        p = subprocess.Popen(' '.join(train_cmd), shell=True)
        return p
    
    def build_train_cmd(self):
        """"""
        nNodes = len(self.nodes)
        np = str(nNodes * int(self.nGPU))
        hosts = ",".join(["{}:{}".format(ip, self.nGPU) for ip in self.nodes])
        cmd = ["mpirun", 
                "-np", np,
                "-H", hosts,
                "-bind-to", "none",
                "-map-by", "slot",
                "-x", "NCCL_DEBUG=INFO",
                "-x", "LD_LIBRARY_PATH",
                "-x", "PATH",
                "-x", "PYTHONPATH={}".format(
                    expanduser("~/autorun/horovod-modified/build/lib.linux-x86_64-3.6")),
                "-mca", "btl ^openib",
                "-mca", "btl_tcp_if_exclude lo,docker0",
                self.python_bin, self.script_path, 
                self.script_args]
        return cmd
    
    def _get_logs(self):
        cpu_logs, net_logs = self._get_cpu_net_log()
        hook_logs, model_logs, mpi_logs = self._get_horovod_logs()
        return cpu_logs, net_logs, hook_logs, model_logs, mpi_logs
    
    def run(self):
        """"""
        self._init_host_env()

        print('='*10, "working on bandwidth control")
        self.bandwith_control()
        print('='*10, "bandwidth control DONE")

        cpu_p, net_p = self._exe_res_monitor()
        print(">"*10, 'launched CPU & Network monitoring')

        print('*'*10, 'Start working on experiment script')
        train_p = self.exe_dist_train()
        train_p.wait()
        print('*'*10, 'Experiment finished')

        cpu_p.terminate()
        net_p.terminate()

        print('End experiment')
        self.move_log()

    def _exe_res_monitor(self):
        """ execute cpu and network bandwidth monitor
        """
        # record existing logs
        cpu_monitor_script = expanduser("~/autorun/monitor_cpu.py")
        net_monitor_script = expanduser("~/autorun/monitor_net.py")
        cpu_p = subprocess.Popen([self.python_bin, cpu_monitor_script],
            stdout=subprocess.DEVNULL)
        net_p = subprocess.Popen([self.python_bin, net_monitor_script],
            stdout=subprocess.DEVNULL)
        return cpu_p, net_p

    def move_log(self):
        """ rename horovod_logs -> horovod_logs_<bandwidth>,
        moving cpu.log and net.log into horovod_logs_<bandwidth> folder
        """
        # cpu, net, hook, model, mpi
        n_cpu, n_net, n_hook, n_model, n_mpi = self._get_logs()
        e_cpu, e_net, e_hook, e_model, e_mpi = self.exist_logs
        def _moving(src, dst, files):
            for _f in files:
                shutil.copy2(join(src, _f), join(dst, _f))
        dst_folder = self.log_folder if self.log_folder \
            else "./log_archives/{}-{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 
                self.bw_limit)
        os.makedirs(dst_folder)
        _moving("./logs/cpu", dst_folder, n_cpu - e_cpu)
        _moving("./logs/net", dst_folder, n_net - e_net)
        _moving(expanduser("~/horovod_logs/hooks"), dst_folder, n_hook-e_hook)
        _moving(expanduser("~/horovod_logs/model_log/"), dst_folder, n_model-e_model)
        _moving(expanduser("~/horovod_logs/mpi_events"), dst_folder, n_mpi-e_mpi)
        with open(join(dst_folder, "readme"), 'w+') as ofile:
            ofile.write("bandwidth limit: " + self.bw_limit)
    
    def _get_cpu_net_log(self):
        """ 
        record current exisiting logs
        """
        log_path = "./logs"
        log_path = expanduser(log_path)
        net_logs = os.listdir(join(log_path, 'net'))
        cpu_logs = os.listdir(join(log_path, 'cpu'))
        return set(cpu_logs), set(net_logs)

    def _get_horovod_logs(self):
        base_dir = expanduser("~/horovod_logs")
        hook_logs = os.listdir(join(base_dir, "hooks"))
        model_logs = os.listdir(join(base_dir, "model_log"))
        mpi_logs = os.listdir(join(base_dir, "mpi_events"))
        return set(hook_logs), set(model_logs), set(mpi_logs)


def main():
    """"""
    python_bin = "/usr/bin/python3"
    exp = ExpRunner(python_bin, 
                "~/autorun/distributed-training/test_scripts/pytorch_resnet101_cifar10.py", 
                "--epochs 1", # args of the script we want to run
                ["localhost", "172.31.24.153"], # list of worker's ip
                nGPU="1", # nGPU on each machine
                eth="ens3", # NIC interface name, used for bandwidth limit
                bw_limit="", # limiting bandwidth, 100Mbit, 1Gbit, 10Gbit 25Gbit, 40Gbit,
                log_folder="" # if not specified, it will used the timestamp
                )
    exp.run()


if __name__ == "__main__":
    main()