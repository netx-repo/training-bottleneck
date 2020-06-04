import subprocess
import os
from os.path import expanduser
import shlex


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

if __name__ == "__main__":
    main()