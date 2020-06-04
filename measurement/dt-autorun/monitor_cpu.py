import psutil
import time
from datetime import datetime
import os
import sys


def create_cpu_logfile():
    log_folder = "./logs/cpu"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    dt = datetime.fromtimestamp(time.time())
    timestamp = dt.strftime("%Y%m%d-%H%M%S")
    logfile = timestamp + '-cpu.log'
    return os.path.join(log_folder, logfile)

def main():
    cpu_interval = 1

    if len(sys.argv) > 1:
        # second arg as output file path
        logfile = sys.argv[1]
    else:
        logfile = create_cpu_logfile()
    with open(logfile, 'w') as cpu_log:
        
        while True:
            cpu_percent = psutil.cpu_percent(interval=cpu_interval, percpu=True)
            print("{}, {}".format(time.time(), cpu_percent))
            cpu_log.write("{}, {}\n".format(time.time(), 
                cpu_percent))
            cpu_log.flush()


if __name__ == "__main__":
    main()