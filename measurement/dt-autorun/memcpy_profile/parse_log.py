import os
import json
from os.path import join, basename

def parse(logpath):
    DtoH = ""
    HtoD = ""
    with open(logpath) as ifile:
        for line in ifile:
            if "[CUDA memcpy DtoH]" in line:
                DtoH = line
            if "[CUDA memcpy HtoD]" in line:
                HtoD = line
    def _parse_line(line_str):
        segs = line_str.split()
        if len(segs) < 6:
            print(segs)
        avg_time = line_str.split()[-6]

        return avg_time
    if DtoH == "" or HtoD == "":
        print(logpath)
        return (None, None)
    return _parse_line(DtoH), _parse_line(HtoD)


def parse_logs(path, collector):
    if os.path.isdir(path):
        files = os.listdir(path)
        for f in files:
            parse_logs(join(path, f), collector)
        return
    
    size = basename(path)
    avgD2H, avgH2D = parse(path)
    collector[int(size)] = {"DtoH(avg)": avgD2H, "HtoD(avg)": avgH2D}
    return 


def main():
    """"""
    
    # assume single file first
    log_folder = "./profile_logs"
    avg_time_collector = {}
    parse_logs(log_folder, avg_time_collector)
    print(avg_time_collector)
    with open("./summary.json", "w") as ofile:
        json.dump(avg_time_collector, ofile, indent=2)
    
if __name__ == "__main__":
    main()