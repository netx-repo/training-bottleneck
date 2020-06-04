import sys, json
import subprocess
from docker_dt import ExpRunner
from os.path import expanduser
from pssh.clients.native.single import SSHClient

class SingleNodeExp(ExpRunner):

    def __init__(self, config):
        """"""
        self.config = config
        self._parse_config(config)
    
    def _parse_config(self, config):
        self.host_user_dir = config["host_user_dir"]
        self.docker_user_dir = config["docker_user_dir"]
        self.docker_user = config["docker_user"]
        self.docker_ssh_port = config["docker_ssh_port"]
        self.script_path = self._trans_docker_path(config["script_path"])
        self.script_args = config["script_args"]
        self.log_folder = config['log_folder']
        self.docker_key = config["docker_ssh_key"]
        self.bw_limit = "ST"
        self.default_bw = "ST"
    
    def _start_containers(self):
        """ start local container only"""
        stop_cmd = "docker kill $(docker ps -q)"
        pull_cmd = "docker pull zarzen/horovod-mod:1.0"

        start_cmd = "sudo docker run --gpus 1 --network=host --detach --ipc=host "\
            "-v {}/autorun/distributed-training:{}/distributed-training "\
            "-v {}/autorun/horovod_logs:{}/horovod_logs "\
            "-v {}/data:{}/data "\
            "zarzen/horovod-mod:1.0".format(self.host_user_dir, self.docker_user_dir,
                                            self.host_user_dir, self.docker_user_dir,
                                            self.host_user_dir, self.docker_user_dir)
        subprocess.run(stop_cmd, shell=True)
        subprocess.run(pull_cmd, shell=True)
        subprocess.run(start_cmd, shell=True)
    
    def _init_host_env(self):
        check_cmd = "mkdir ~/autorun; mkdir ~/autorun/horovod_logs; " \
                        "mkdir ~/autorun/horovod_logs/hooks; "\
                        "mkdir ~/autorun/horovod_logs/model_log; "\
                        "mkdir ~/autorun/horovod_logs/mpi_events; "\
                        "mkdir ~/autorun/logs/; "\
                        "mkdir ~/autorun/logs/net; mkdir ~/autorun/logs/cpu; mkdir ~/data "
        subprocess.run(check_cmd, shell=True)

        check_cmd = "cd ~/autorun; ls|grep distributed-training"
        output = subprocess.check_output(check_cmd, shell=True)
        if output != b"":
            git_pull = "cd ~/autorun/distributed-training; git pull"
            subprocess.run(git_pull, shell=True)
        else:
            cmd = "cd ~/autorun;"\
                    "git clone https://github.com/zarzen/distributed-training.git"
            subprocess.run(cmd, shell=True) 
    
    def run(self):
        """"""
        self._init_host_env()
        self._start_containers()
        self._init_docker_ssh()

        self.exist_logs = self._get_logs()
        # self._exe_cmd(self.contianer, "ls")
        cmd = self.build_train_cmd()
        print('running command:', cmd)
        self._exe_cmd(self.contianer, cmd)

        print('End experiment')
        self.move_log()

    def build_train_cmd(self):
        """"""
        exp_cmd = "python3 {} {}".format(self.script_path, self.script_args)
        return exp_cmd

    def _init_docker_ssh(self):
        self.contianer = SSHClient("localhost", user=self.docker_user, port=2022,
            pkey=self.docker_key)
    
    def _kill_containers(self):
        stop_cmd = "docker kill $(docker ps -q)" 
        subprocess.run(stop_cmd, shell=True)

    def _exe_cmd(self, client, cmd):
        _channel, _host, stdout, stderr, stdin = client.run_command(cmd)
        for line in stdout:
            print(_host, ":", line)
        for line in stderr:
            print(_host, ":", line)

def main():
    if len(sys.argv) < 2:
        print("Please specific config file")
        sys.exit()
        return 

    with open(sys.argv[1]) as config_file:
        config = json.load(config_file)
        exp = SingleNodeExp(config)
        exp.run()
    

if __name__ == "__main__":
    main()