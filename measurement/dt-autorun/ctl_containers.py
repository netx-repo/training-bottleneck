import paramiko
from os.path import join, expanduser
import sys
import json

class Controller:
    def __init__(self, config):
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

    def _exec_cli_cmd(self, cli, cmd, msg=None):
        if msg:
            print('>'*10, msg, '<'*10)
        _, stdout, stderr = cli.exec_command(cmd)
        print('cmd stdout: ', stdout.read().decode('utf-8'),
              "cmd stderr: ", stderr.read().decode('utf-8'))
        if msg:
            print('>'*10, 'DONE', msg, '<'*10)
    
    def start_containers(self):
        start_cmd = "docker run --gpus all --network=host --detach --ipc=host "\
            "-v {}/autorun/distributed-training:{}/distributed-training "\
            "-v {}/autorun/horovod_logs:{}/horovod_logs "\
            "zarzen/horovod-mod:1.0".format(self.host_user_dir, self.docker_user_dir,
                                            self.host_user_dir, self.docker_user_dir
                                            )
        for (ip, cli) in self.host_nodes:
            self._exec_cli_cmd(cli, start_cmd, "{} start containers".format(ip))
    
    def stop_containers(self):
        stop_cmd = "docker kill $(docker ps -q)"
        for (ip, cli) in self.host_nodes:
            self._exec_cli_cmd(cli, stop_cmd, "{}: stop all containers".format(ip))
    
    def update_containers(self):
        pull_cmd = "docker pull zarzen/horovod-mod:1.0"
        for (ip, cli) in self.host_nodes: 
           self._exec_cli_cmd(cli, pull_cmd, "{}: pull docker image".format(ip))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('specfy config.json and use command : start/stop/update')
        sys.exit()
    
    with open(sys.argv[1]) as ifile:
        config = json.load(ifile)
    ctl = Controller(config)

    if sys.argv[2] == "start":
        ctl.start_containers()
    elif sys.argv[2] == "stop":
        ctl.stop_containers()
    elif sys.argv[2] == "update":
        ctl.update_containers()
    else:
        print('wrong command')
