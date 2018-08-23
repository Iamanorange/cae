#!/usr/bin/env python3

import argparse
import os
from PIL import Image


def main(args):
    path = args.dataset_path
    files= os.listdir(path)
    for f in files:
        if not os.path.isdir(f):
            im = Image.open(path + '/' + f)
            print("resize " + path + '/' + f)
            im = im.resize((1280, 768))
            im = im.convert('RGB')
            im.save(path + '/' + f)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str, required=True)
    args = p.parse_args()
    main(args)
