""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from conf import settings


def get_network(args):
    """ return given network
    """

    if args.cifar not in [10, 100]:
        print('cifar type has to be 10 or 100')
        sys.exit()

    num_classes = args.cifar

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes=num_classes)
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_classes=num_classes)
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes=num_classes)
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_classes=num_classes)
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(num_classes=num_classes)
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(num_classes=num_classes)
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(num_classes=num_classes)
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(num_classes=num_classes)
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet(num_classes=num_classes)
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3(num_classes=num_classes)
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4(num_classes=num_classes)
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2(num_classes=num_classes)
    elif args.net == 'xception':
        from models.xception import xception
        net = xception(num_classes=num_classes)
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=num_classes)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=num_classes)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=num_classes)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=num_classes)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_classes=num_classes)
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18(num_classes=num_classes)
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34(num_classes=num_classes)
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50(num_classes=num_classes)
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101(num_classes=num_classes)
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152(num_classes=num_classes)
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(num_classes=num_classes)
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101(num_classes=num_classes)
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152(num_classes=num_classes)
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet(num_classes=num_classes)
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(num_classes=num_classes)
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet(num_classes=num_classes)
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(num_classes=num_classes)
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(num_classes=num_classes)
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet(num_classes=num_classes)
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56(num_classes=num_classes)
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92(num_classes=num_classes)
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18(num_classes=num_classes)
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34(num_classes=num_classes)
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50(num_classes=num_classes)
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101(num_classes=num_classes)
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152(num_classes=num_classes)
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet(num_classes=num_classes)
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18(num_classes=num_classes)
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34(num_classes=num_classes)
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50(num_classes=num_classes)
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101(num_classes=num_classes)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.device != 'cpu':
        net = net.to(torch.device(args.device))

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_train_val_split_dataloader(val_count=0, existing_train_val_split=False, cifar_type=100,
                                   batch_size=16, num_workers=2, shuffle=True, for_testing=False):
    """ return training dataloader
    Args:
        val_count: number of samples in validation set
        existing_train_val_split: whether to use existing train/val split.
        location of the split indexes is specified in settings
        cifar_type: cifar dataset type (10 or 100)
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle before split
        for_testing: whether to disable the use of random transformations on train_data_loader
    Returns: train_data_loader, val_data_loader: torch dataloader objects.
    In default settings train_data_loader performs random crop, random horizontal flip and random rotation,
    this can be disabled by setting for_testing=True.
    """

    if cifar_type == 100:
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif cifar_type == 10:
        mean = settings.CIFAR10_TRAIN_MEAN
        std = settings.CIFAR10_TRAIN_STD
    else:
        print('Unsupported cifar type')
        sys.exit()

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if cifar_type == 100:
        cifar_training_train_tf = torchvision.datasets.CIFAR100(root='../data', train=True, download=True,
                                                                transform=transform_train)
        cifar_training_val_tf = torchvision.datasets.CIFAR100(root='../data', train=True, download=True,
                                                                transform=transform_val)
    elif cifar_type == 10:
        cifar_training_train_tf = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                                                transform=transform_train)
        cifar_training_val_tf = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                                              transform=transform_val)
    else:
        print('Unsupported cifar type')
        sys.exit()

    if val_count > 0:
        if existing_train_val_split:
            train_idx = np.load(os.path.join(settings.SPLIT_PATH, 'train_idx.npy'))
            val_idx = np.load(os.path.join(settings.SPLIT_PATH, 'val_idx.npy'))
        else:
            train_targets = cifar_training_train_tf.targets
            full_train_size = len(train_targets)
            test_portion = val_count / full_train_size
            train_idx, val_idx = train_test_split(np.arange(full_train_size), test_size=test_portion, shuffle=True,
                                                  stratify=train_targets)

    else:
        train_idx = np.arange(len(cifar_training_train_tf.targets))
        val_idx = np.array([])

    if not os.path.exists(settings.SPLIT_PATH):
        os.mkdir(settings.SPLIT_PATH)

    np.save(os.path.join(settings.SPLIT_PATH, 'train_idx.npy'), np.array(train_idx))
    np.save(os.path.join(settings.SPLIT_PATH, 'val_idx.npy'), np.array(val_idx))

    if not for_testing:
        train_subset = torch.utils.data.Subset(cifar_training_train_tf, train_idx)
    else:
        train_subset = torch.utils.data.Subset(cifar_training_val_tf, train_idx)

    val_subset = torch.utils.data.Subset(cifar_training_val_tf, val_idx)

    cifar_train_loader = DataLoader(
        train_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True
    )

    cifar_val_loader = DataLoader(
        val_subset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=True
    )

    return cifar_train_loader, cifar_val_loader


def get_test_dataloader_general(cifar_type=100, batch_size=16, num_workers=2, shuffle=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    if cifar_type == 100:
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif cifar_type == 10:
        mean = settings.CIFAR10_TRAIN_MEAN
        std = settings.CIFAR10_TRAIN_STD
    else:
        print('Unsupported cifar type')
        sys.exit()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    if cifar_type == 100:
        cifar_test = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    elif cifar_type == 10:
        cifar_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    else:
        print('Unsupported cifar type')
        sys.exit()

    cifar_test_loader = DataLoader(
        cifar_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar_test_loader

def get_test_dataloader(mean, std, cifar_type=100, batch_size=16, num_workers=2, shuffle=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([cifar100_dataset[i][0][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][0][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][0][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
