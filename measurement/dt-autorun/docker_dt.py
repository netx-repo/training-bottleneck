import os
import sys
import json
from os.path import expanduser, join, abspath
import subprocess
import datetime
import shutil
import paramiko


class ExpRunner:

    def __init__(self, config) -> None:
        """"""
        self.config = config
        self._config_parser(self.config)

        self._init_host_ssh()
    
    def _config_parser(self, config):
        """ parse json object
        """
        self.host_user_dir = config["host_user_dir"]
        self.docker_user_dir = config["docker_user_dir"]
        self.docker_user = config["docker_user"]
        self.docker_ssh_port = config["docker_ssh_port"]
        self.script_path = self._trans_docker_path(config["script_path"])
        self.script_args = config["script_args"]
        self.nodes = config['nodes']
        self.nGPU = config['nGPU'] # for each machine
        self.eth = config['eth'] # name if NIC
        self.bw_limit = config['bw_limit']
        self.default_bw = config['default_bw']
        self.log_folder = config['log_folder']
        self.host_key = paramiko.RSAKey.from_private_key_file(expanduser(config["host_ssh_key"]))
        self.docker_key = paramiko.RSAKey.from_private_key_file(config["docker_ssh_key"])
        
    def _trans_docker_path(self, path):
        return path.replace('~', self.docker_user_dir)

    def _init_host_ssh(self):
        print('='*10, 'initializing ssh connections')
        self.host_nodes = []
        for node in self.nodes:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=node, username="ubuntu", pkey=self.host_key)
            self.host_nodes.append((node, client))
            print('IP', node, 'DONE')
        print('='*10, 'initialization for ssh host node DONE')
    
    def _init_host_env(self):
        """"""
        for ip, cli in self.host_nodes:
            check_cmd = "mkdir ~/autorun; mkdir ~/autorun/horovod_logs; " \
                        "mkdir ~/autorun/horovod_logs/hooks; "\
                        "mkdir ~/autorun/horovod_logs/model_log; "\
                        "mkdir ~/autorun/horovod_logs/mpi_events; "\
                        "mkdir ~/autorun/logs/; "\
                        "mkdir ~/autorun/logs/net; mkdir ~/autorun/logs/cpu; mkdir ~/data "
            self._exec_cli_cmd(cli, check_cmd)
            check_cmd = "cd ~/autorun; ls|grep distributed-training"
            _, stdout, stderr = cli.exec_command(check_cmd)
            if stdout.read() != b"":
                git_pull = "cd ~/autorun/distributed-training; git pull"
                self._exec_cli_cmd(cli, git_pull, '{}: git pull'.format(ip))
            else:
                cmd = "cd ~/autorun;"\
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

    def _start_containers(self):
        """"""
        stop_cmd = "docker kill $(docker ps -q)"
        pull_cmd = "docker pull zarzen/horovod-mod:1.0"

        start_cmd = "sudo docker run --gpus all --network=host --detach --ipc=host "\
            "-v {}/autorun/distributed-training:{}/distributed-training "\
            "-v {}/autorun/horovod_logs:{}/horovod_logs "\
            "-v {}/data:{}/data "\
            "zarzen/horovod-mod:1.0".format(self.host_user_dir, self.docker_user_dir,
                                            self.host_user_dir, self.docker_user_dir,
                                            self.host_user_dir, self.docker_user_dir)
        self.docker_ids = {}
        for (ip, cli) in self.host_nodes:
            print('>'*10, ip, '<'*10)
            self._exec_cli_cmd(cli, stop_cmd, "{}: stop all containers".format(ip))
            self._exec_cli_cmd(cli, pull_cmd, "{}: pull docker image".format(ip))
            _, stdout, stderr = cli.exec_command(start_cmd)
            _docker_id = stdout.read().decode('utf-8')
            self.docker_ids[ip] = _docker_id
            print('docker_id', _docker_id)
            print('Start Errors:', stderr.read().decode('utf-8'))
            print('='*10, ip, 'start container DONE', '='*10)

    def _kill_containers(self):
        """ after experiments done"""
        print('*'*10, 'killing docker containers')
        kill_cmd = "docker container kill {}"
        for ip, cli in self.host_nodes:
            if ip in self.docker_ids:
                self._exec_cli_cmd(cli, kill_cmd.format(self.docker_ids[ip]), ip)
        print('*'*10, 'kill containers done')

    def bandwith_control(self):
        """
        """
        del_cmd = "sudo tc qdisc del dev {} root tbf rate 40Gbit latency 400ms burst 3000kbit".format(self.eth)
        # if self.bw_limit = "" then we don't execute the add_cmd
        add_cmd = "sudo tc qdisc add dev {} root tbf rate {} latency 400ms burst 3000kbit".format(self.eth, self.bw_limit)
        for (ip, cli) in self.host_nodes:
            # try to delete rate limit
            self._exec_cli_cmd(cli, del_cmd, "{}: delete bandwidth limit".format(ip))
            # ensure limit deleted
            self._exec_cli_cmd(cli, del_cmd, "{}: delete bandwidth limit".format(ip)) 
            if self.bw_limit:
                self._exec_cli_cmd(cli, add_cmd, "{}: add bandwidth limit {}".format(ip, self.bw_limit))

    def exec_dist_train(self):
        """ execute distributed training script at rank0
        :return process:
        """
        train_cmd = self.build_train_cmd()
        print("Exec:", " ".join(train_cmd))

        # ssh into rank0 container
        ip, _ = self.host_nodes[0]
        rank0 = paramiko.SSHClient()
        rank0.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        rank0.connect(hostname=ip, port=self.docker_ssh_port, 
                        username=self.docker_user, 
                        pkey=self.docker_key) 

        def line_buffered(f):
            line_buf = ""
            while not f.channel.exit_status_ready():
                c = f.read(1).decode('utf-8') 
                if c != '\n':
                    line_buf += c 
                else:
                    yield line_buf
                    line_buf = ''
        
        _, stdout, stderr = rank0.exec_command(" ".join(train_cmd), bufsize=100)
        print("-"*10, 'training log')
        for line in line_buffered(stdout):
            print(line)
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))
        print('-'*10, 'training log end')
    
    def build_train_cmd(self):
        """"""
        nNodes = len(self.nodes)
        np = str(nNodes * int(self.nGPU))
        hosts = ",".join(["{}:{}".format(ip, self.nGPU) for ip in self.nodes])
        cmd = ["NCCL_DEBUG=INFO",
                "HOROVOD_NUM_NCCL_STREAMS=4",
                "horovodrun", 
                "-np", np,
                "-H", hosts,
                "python3", 
                self.script_path, 
                self.script_args,
                "|& grep -v \"Read -1\""]
        return cmd
    
    def _get_logs(self):
        cpu_logs, net_logs = self._get_cpu_net_log()
        hook_logs, model_logs, mpi_logs = self._get_horovod_logs()
        return cpu_logs, net_logs, hook_logs, model_logs, mpi_logs
    
    def run(self):
        """"""

        print('initiating host env')
        self._init_host_env()

        self.exist_logs = self._get_logs()
        print('='*10, "working on bandwidth control")
        self.bandwith_control()
        print('='*10, "bandwidth control DONE")

        cpu_p, net_p = self._exe_res_monitor()
        print(">"*10, 'launched CPU & Network monitoring')

        print('='*10, 'Start containers', )
        self._start_containers()

        print('*'*10, 'Start working on experiment script')
        self.exec_dist_train()
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
        cpu_p = subprocess.Popen(["python3", cpu_monitor_script],
            stdout=subprocess.DEVNULL)
        net_p = subprocess.Popen(["python3", net_monitor_script],
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
        dst_folder = "./log_archives/{}-{}-{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 
                self.bw_limit, self.default_bw)
        if self.log_folder:
            dst_folder += '-' + self.log_folder

        os.makedirs(dst_folder)
        _moving("./logs/cpu", dst_folder, n_cpu - e_cpu)
        _moving("./logs/net", dst_folder, n_net - e_net)
        _moving("./horovod_logs/hooks", dst_folder, n_hook-e_hook)
        _moving("./horovod_logs/model_log/", dst_folder, n_model-e_model)
        _moving("./horovod_logs/mpi_events", dst_folder, n_mpi-e_mpi)
        with open(join(dst_folder, "readme.txt"), 'w+') as ofile:
            ofile.write("bandwidth limit: {}\n".format(self.bw_limit))
            train_cmd = self.build_train_cmd()
            ofile.write("execute cmd: {}\n".format(" ".join(train_cmd)))
        with open(join(dst_folder, "config.json"), 'w') as ofile:
            json.dump(self.config, ofile, indent=4)
    
    def _get_cpu_net_log(self):
        """ 
        record current exisiting logs
        """
        log_path = "./logs"
        log_path = expanduser(log_path)
        net_logs = os.listdir(join(log_path, 'net'))
        cpu_logs = os.listdir(join(log_path, 'cpu'))
        return set(cpu_logs), set(net_logs)
    
    def _create_horovod_logs_folder(self):
        base_dir = "./horovod_logs"
        if not os.path.exists(base_dir):
            os.makedirs('./horovod_logs')
        if not os.path.exists(join(base_dir, "hooks")):
            os.makedirs(join(base_dir, "hooks"))
        if not os.path.exists(join(base_dir, "model_log")):
            os.makedirs(join(base_dir, "model_log"))
        if not os.path.exists(join(base_dir, "mpi_events")):
            os.makedirs(join(base_dir, "mpi_events"))

    def _get_horovod_logs(self):
        base_dir = "./horovod_logs"
        hook_logs = os.listdir(join(base_dir, "hooks"))
        model_logs = os.listdir(join(base_dir, "model_log"))
        mpi_logs = os.listdir(join(base_dir, "mpi_events"))
        return set(hook_logs), set(model_logs), set(mpi_logs)

    def __del__(self):
        self._kill_containers()

def main():
    """"""
    if len(sys.argv) < 2:
        print("Please specific config file")
        sys.exit()
        return 
    with open(sys.argv[1]) as config_file:
        config = json.load(config_file)
        exp = ExpRunner(config)
        exp.run()


if __name__ == "__main__":
    main()