import os
import sys
import json

nodes = []
def change_nodes(filepath):
    with open(filepath,'r') as f:
        configs = json.load(f)
    configs['nodes'] = nodes
    with open(filepath,'w') as f:
        json.dump(configs,f, indent=4)

def travelPath(rootDir):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if lists[0]=='.':
            continue
        if os.path.isfile(path):
            if path.split('.')[-1] == 'json':
                print(path)
                change_nodes(path)
        if os.path.isdir(path):
            travelPath(path)

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("You need input the config's path and host's ip")
    elif len(sys.argv[1].split('.')) == 4:
        print("Your first input must be the config's path")
    else:
        host_num = len(sys.argv)-2
        print("number of nodes: ", host_num)
        for i,node in enumerate(sys.argv):
            if i < 2:
                continue
            print(node)
            nodes.append(node)
    print(nodes)
    travelPath(sys.argv[1])
