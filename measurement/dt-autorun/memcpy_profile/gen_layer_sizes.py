from torchvision import models as m
import os
from os.path import join

def main():
    """"""
    models = {
        "resnet50": m.resnet50,
        "resnet101": m.resnet101,
        "vgg16": m.vgg16_bn
    }
    folder = "./model_sizes"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for m_name in models:
        model = models[m_name]()
        mlayers = model.state_dict()
        layer_sizes = []
        for lname in mlayers:
            n = 1
            for s in mlayers[lname].size():
                n *= s
            layer_sizes += [n]
        with open(join(folder, "{}.txt".format(m_name)), "w") as ofile:
            for n in layer_sizes:
                ofile.write("{}\n".format(n))

if __name__ == "__main__":
    main()