import json

def main():
    """"""
    with open("./template.json") as ifile:
        template_config = json.load(ifile)
    
    models = ["resnet50", "resnet101", "vgg16"]
    datas = ["cifar10", "imagenet"]
    for m in models:
        for d in datas:
            script = "~/distributed-training/test_scripts/single_node/pytorch/{}_{}.py".format(
                m, d
            )
            log_folder = "{}-{}-ST".format(m, d)
            template_config["script_path"] = script
            template_config['log_folder'] = log_folder

            with open("{}-{}.json".format(m, d), 'w') as ofile:
                json.dump(template_config, ofile, indent=2)


if __name__ == "__main__":
    main()