import subprocess

def main():
    """"""
    config_t = "./training-configs/single_node/{}-{}.json"
    models = ["resnet50", "resnet101", "vgg16"]
    datas = ["cifar10", "imagenet"]

    for m in models:
        for d in datas:
            print(">"*10, "running experiment for", m, d)
            config = config_t.format(m, d)
            cmd = "python3 docker_st.py {}".format(config)
            subprocess.run(cmd, shell=True)
            print("<"*10, m, d, "done")

if __name__ == "__main__":
    main()