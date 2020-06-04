import subprocess
import os
from os.path import join, exists

def main():
    """"""
    folder = "./model_sizes"
    files = ["resnet50.txt", "resnet101.txt", "vgg16.txt"]
    prof_folder = "./profile_logs"
    if not exists(prof_folder):
        os.makedirs(prof_folder)
    
    for f in files:
        if not exists(join(prof_folder, f)):
            os.makedirs(join(prof_folder, f))
        with open(join(folder, f)) as ifile:
            for line in ifile:
                size = int(line)
                output_log = join(prof_folder, f, str(size))
                cmd = "sudo nvprof --log-file {} ./profile {} 20".format(
                    output_log, size
                )
                subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()