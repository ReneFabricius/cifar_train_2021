import argparse
import os
import torch

from train_ja import train_script


def main():
    networks = ['seresnet34', 'stochasticdepth50', 'googlenet', 'resnext101', 'inceptionv3',
                'densenet121', 'resnet34', 'xception']

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-repl', type=int, default=1, help='number of replications')
    parser.add_argument('-batch_sz', type=int, default=64, help='batch size')
    parser.add_argument('-device', type=str, default="cpu", help='Device on which to perform the computations')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-val_size', default=0, type=int, help='number of images in validation set')
    parser.add_argument('-num_net', default=3, type=int,
                        help='Number of network architectures to train (from a fixed list)')
    parser.add_argument('-architectures', default=[], nargs='+', help="List of architectures to use, overrides num_net")
    parser.add_argument('-output_ood', action='store_true', dest='output_ood', help="Whether to produce ood outputs (the other cifar)")
    parser.add_argument('-cifar_data', default="", help="Cifar data folder")
    args = parser.parse_args()

    os.chdir(args.folder)

    for repli in range(args.repl):
        repl_dir = os.path.join(args.folder, str(repli))
        print('Replication {}'.format(repli))
        if not os.path.exists(repl_dir):
            os.mkdir(repl_dir)

        os.chdir(repl_dir)
        
        net_archs = args.architectures
        if len(net_archs) == 0:
            net_archs = networks[:args.num_net]
        for i, arch in enumerate(net_archs):
            print('Processing architecture {}'.format(arch))
            fin = False
            tries = 0
            cur_b = args.batch_sz
            while not fin and tries < 20 and cur_b > 0:
                if tries > 0:
                    torch.cuda.empty_cache()
                    print('Trying again, try {}, batch size {}'.format(tries, cur_b))
                try:
                    if i == 0:
                        train_script(net=arch, device=args.device, cifar=args.cifar, val_split_size=args.val_size,
                                     b=cur_b, output_ood=args.output_ood, cifar_data=args.cifar_data)
                    else:
                        train_script(net=arch, device=args.device, cifar=args.cifar, val_split_size=args.val_size,
                                     val_split_existing=True, b=cur_b, output_ood=args.output_ood, cifar_data=args.cifar_data)
                    fin = True
                except RuntimeError as rerr:
                    if 'memory' not in str(rerr):
                        raise rerr
                    print("OOM Exception")
                    del rerr
                    cur_b = int(0.9 * cur_b)
                    tries += 1

            if not fin:
                print('Unsuccessful')
                return -1

        os.chdir('../')


if __name__ == '__main__':
    main()
