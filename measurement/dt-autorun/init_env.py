"""
build horovod-modified
"""
import os
import sys
from os.path import expanduser, join
import shlex
import subprocess
import paramiko

class Initializer:
    def __init__(self, IPs, nccl_home) -> None:
        """
        """
        self.IPs= IPs
        self.nccl_home = nccl_home

        self._init_ssh()

    def _init_ssh(self):
        key = paramiko.RSAKey.from_private_key_file(
            expanduser("~/.ssh/id_rsa"))
        print('='*10, 'initializing ssh connections')
        self.clients = []
        for node in self.IPs:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=node, username="ubuntu", pkey=key)
            self.clients.append((node, client))
            print('IP', node, 'DONE')
        print('='*10, 'initialization for ssh clients DONE')
    
    def init(self):
        self.download()
        self.build()
        self.append_PYTHONPATH()
    
    def download(self):
        """"""
        for ip, cli in self.clients:
            check_cmd = "mkdir autorun; cd ~/autorun; ls|grep horovod-modified"
            _, stdout, stderr = cli.exec_command(check_cmd)
            if stdout.read() != b'':
                print(ip, 'source file exist')
            else:
                cmd = "cd ~/autorun; "\
                        "wget https://dt-training.s3.amazonaws.com/horovod-modified.tar.gz;" \
                        "tar zxf horovod-modified.tar.gz; "\
                        "rm horovod-modified.tar.gz"
                _, stdout, stderr = cli.exec_command(cmd)
                print(ip, ":", "stdout::", stdout.read(), "stderr::", stderr.read())

            check_cmd = "cd ~/autorun; ls|grep distributed-training"
            _, stdout, stderr = cli.exec_command(check_cmd)
            if stdout.read() != b"":
                print(ip, "distributed-training folder exisit")
            else:
                cmd = "cd ~/autorun;"\
                    "git clone https://github.com/zarzen/distributed-training.git"
                _, stdout, stderr = cli.exec_command(cmd)
                print(ip, '')
        
    def build(self):
        """"""
        for ip, cli in self.clients:
            print("-"*10, 'horovod building at', ip)
            cmd = "cd ~/autorun/horovod-modified; "\
                "HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=" \
                    + self.nccl_home + " python3 setup.py build" 
            _, stdout, stderr = cli.exec_command(cmd)
            print('stdout::', stdout.read(), 'stderr::', stderr.read())
            print('-'*10, 'horovod built at', ip)

    def append_PYTHONPATH(self):
        """
        """
        for ip, cli in self.clients:
            check_cmd = "echo $PYTHONPATH | "\
                "grep autorun/horovod-modified/build/lib.linux-x86_64-3.6"
            _, stdout, stderr = cli.exec_command(check_cmd)
            if stdout.read() != b"":
                print('PYTHONPATH exists')
            else:
                cmd = "echo 'export PYTHONPATH=\"{}\":$PYTHONPATH' >> ~/.bashrc".format(
                    join(expanduser("~/autorun/horovod-modified"), "build/lib.linux-x86_64-3.6")
                )
                _, stdout, stderr = cli.exec_command(cmd)
                print(ip, ": append PYTHONPATH")

    def delete(self):
        """
        delete source for fresh download
        """
        for ip, cli in self.clients:
            print('>'*10, 'deleting source files at ', ip)
            cmd = "rm -rf ~/autorun/horovod-modified/; rm -rf ~/autorun/distributed-training"
            _, stdout, stderr = cli.exec_command(cmd)
            print(ip, 'stdout::', stdout.read(), 'stderr::', stderr.read())
            print(ip, "deleted sources")
    
    def update_scripts(self):
        """"""
        for ip, cli in self.clients:
            print('>'*10, 'updating training script')
            cmd = "cd ~/autorun/distributed-training; git pull"
            _, stdout, stderr = cli.exec_command(cmd)
            print(ip, 'stdout::', stdout.read(), 'stderr::', stderr.read())
            print('>'*10, 'training source update done at', ip)

def check_bash_env(horovod_path):
    lib_path = join(str(expanduser(horovod_path)), "build/lib.linux-x86_64-3.6")
    PYTHONPATH = os.getenv("PYTHONPATH")
    print(PYTHONPATH)
    if not PYTHONPATH or lib_path not in PYTHONPATH:
        cmd = "echo 'export PYTHONPATH=\"{}\":$PYTHONPATH' >> ~/.bashrc".format(lib_path)
        print(cmd)
        subprocess.call(cmd, shell=True, executable='/bin/bash')
        subprocess.call("source ~/.bashrc", shell=True, executable='/bin/bash')



def main():
    """
    HOROVOD_GPU_ALLREDUCE=NCCL
    HOROVOD_NCCL_HOME=
    
    disable TF and MXNET
    HOROVOD_WITH_TENSORFLOW=0 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=0

    """
    nccl_home = "/usr/local/nccl"
    python_bin = "/usr/bin/python3"
    horovod_path = "~/horovod-modified"
    
    horovod_build_cmd = "{} setup.py build".format(python_bin)
    horovod_build_env = {
        "PATH": os.getenv("PATH"),
        "HOROVOD_WITH_PYTORCH": "1",
        "HOROVOD_GPU_ALLREDUCE": "NCCL"
    }
    if nccl_home:
        horovod_build_env['HOROVOD_NCCL_HOME'] = nccl_home
    subprocess.call(shlex.split(horovod_build_cmd), 
        env=horovod_build_env, cwd=expanduser(horovod_path))

    check_bash_env(horovod_path)
    print(">"*10, "need to run `source ~/.bashrc`, to make sure the added PYTHONPATH works")

if __name__ == "__main__":
    # main()
    if len(sys.argv) < 2:
        print("Need command: init/delete/update")
    helper = Initializer(
        IPs = ['localhost', '172.31.29.187'],
        nccl_home = "/usr/local/nccl"
    )

    if sys.argv[1] == "init":
        helper.init()
    elif sys.argv[1] == "delete":
        helper.delete()
    elif sys.argv[1] == 'update':
        helper.update_scripts()
    else:
        print('wrong command')