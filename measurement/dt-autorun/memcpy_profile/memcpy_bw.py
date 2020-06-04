import json

def extract_time(ts):
    t = float(ts[:len(ts)-2])
    if ts.endswith("ms"):
        return t * 1e-3
    if ts.endswith('us'):
        return t * 1e-6
    if ts.endswith('ns'):
        return t * 1e-9

def main():
    """"""
    with open("./summary.json") as ifile:
        data = json.load(ifile)
        for size in data:
            s = int(size) * 4 # 4 bytes per parameter
            t_d2h = extract_time(data[size]['DtoH(avg)'])
            t_h2d = extract_time(data[size]['HtoD(avg)'])
            bw_d2h = s / t_d2h
            bw_h2d = s / t_h2d
            print("Bytes size {}, HtoD bw: {:.2f} GB/s, DtoH bw: {:.2f} GB/s".format(
                s, bw_h2d/1e9, bw_d2h/1e9))


if __name__ == "__main__":
    main()