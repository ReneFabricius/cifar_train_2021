import argparse
import torch
import os

from train_ja import train_script


def main():
    networks = ['densenet121', 'resnet34', 'xception']

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-batch_sz', type=int, default=64, help='batch size')
    parser.add_argument('-device', type=str, required=True, help='Device on which to perform the computations')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-val_size', default=0, type=int, help='number of images in validation set')
    args = parser.parse_args()

    os.chdir(args.folder)

    for repli in range(args.repl):
        print('Replication {}'.format(repli))
        repl_dir = os.path.join(args.folder, str(repli))
        if not os.path.exists(repl_dir):
            os.mkdir(repl_dir)

        os.chdir(repl_dir)

        for i, arch in enumerate(networks):
            print('Processing architecture {}'.format(arch))
            if i == 0:
                train_script(net=arch, device=args.device, cifar=args.cifar, val_split_size=args.val_size, b=args.batch_sz)
            else:
                train_script(net=arch, device=args.device, cifar=args.cifar, val_split_existing=True, b=args.batch_sz)

        os.chdir('../')


if __name__ == '__main__':
    main()