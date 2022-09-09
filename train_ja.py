import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_train_val_split_dataloader, \
    get_test_dataloader_general


def train(epoch, net, optimizer, loss_function, training_loader, warmup_scheduler, writer, args):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):

        if args.device != 'cpu':
            labels = labels.to(torch.device(args.device))
            images = images.to(torch.device(args.device))

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(net, test_loader, loss_function, writer, args, epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if args.device != 'cpu':
            images = images.to(torch.device(args.device))
            labels = labels.to(torch.device(args.device))

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.device != 'cpu':
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(device=torch.device(args.device)), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)

@torch.no_grad()
def produce_outputs(net, args):
    print('Saving outputs')
    recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
    best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
    net.load_state_dict(torch.load(weights_path))
    net.eval()

    train_loader_ordered, val_loader_ordered = get_train_val_split_dataloader(val_count=args.val_split_size,
                                                                              existing_train_val_split=True,
                                                                              cifar_type=args.cifar, shuffle=False,
                                                                              for_testing=True,
                                                                              data_folder=None if args.cifar_data=="" else args.cifar_data)
    test_loader_ordered = get_test_dataloader_general(cifar_type=args.cifar, shuffle=False,
                                                                              data_folder=None if args.cifar_data=="" else args.cifar_data)
    if args.output_ood:
        ood_loader_ordered = get_test_dataloader_general(cifar_type=10 if args.cifar == 100 else 100, shuffle=False,
                                                                              data_folder=None if args.cifar_data=="" else args.cifar_data)

    if not os.path.exists(settings.OUTPUTS_PATH):
        os.mkdir(settings.OUTPUTS_PATH)

    outputs_path = os.path.join(settings.OUTPUTS_PATH, args.net)
    if not os.path.exists(outputs_path):
        os.mkdir(outputs_path)

    train_outputs = []
    train_labels = []
    print('Processing train set')
    for images, labels in train_loader_ordered:

        if args.device != 'cpu':
            images = images.to(torch.device(args.device))

        output = net(images)
        train_outputs.append(output.detach().cpu().clone().numpy())
        train_labels.append(labels.detach().clone().numpy())

    train_out = np.concatenate(train_outputs)
    train_lab = np.concatenate(train_labels)
    np.save(os.path.join(outputs_path, 'train_outputs.npy'), train_out)
    np.save(os.path.join(outputs_path, 'train_labels.npy'), train_lab)

    val_outputs = []
    val_labels = []
    print('Processing validation set')
    for images, labels in val_loader_ordered:

        if args.device != 'cpu':
            images = images.to(torch.device(args.device))

        output = net(images)
        val_outputs.append(output.detach().cpu().clone().numpy())
        val_labels.append(labels.detach().clone().numpy())

    if len(val_loader_ordered.dataset) > 0:
        val_out = np.concatenate(val_outputs)
        val_lab = np.concatenate(val_labels)
        np.save(os.path.join(outputs_path, 'val_outputs.npy'), val_out)
        np.save(os.path.join(outputs_path, 'val_labels.npy'), val_lab)

    test_outputs = []
    test_labels = []
    print('Processing test set')
    for images, labels in test_loader_ordered:

        if args.device != 'cpu':
            images = images.to(torch.device(args.device))

        output = net(images)
        test_outputs.append(output.detach().cpu().clone().numpy())
        test_labels.append(labels.detach().clone().numpy())

    test_out = np.concatenate(test_outputs)
    test_lab = np.concatenate(test_labels)
    np.save(os.path.join(outputs_path, 'test_outputs.npy'), test_out)
    np.save(os.path.join(outputs_path, 'test_labels.npy'), test_lab)

    if args.output_ood:
        ood_outputs = []
        ood_labels = []
        print("Processing ood set")
        for images, labels in ood_loader_ordered:

            if args.device != 'cpu':
                images = images.to(torch.device(args.device))

            output = net(images)
            ood_outputs.append(output.detach().cpu().clone().numpy())
            ood_labels.append(labels.detach().clone().numpy())

        ood_out = np.concatenate(ood_outputs)
        ood_lab = np.concatenate(ood_labels)
        np.save(os.path.join(outputs_path, 'ood_outputs.npy'), ood_out)
        np.save(os.path.join(outputs_path, 'ood_labels.npy'), ood_lab)
       

def train_script(net, device='cpu', b=128, warm=1, lr=0.1, resume=False, cifar=100, val_split_size=0,
                 val_split_existing=False, output_ood=False, cifar_data=""):
    """

    Args:
        net: string specifying network architecture
        device: device (as string) on which to run the script
        b: batch size
        warm: number of epochs to do warm up for
        lr: starting learning rate
        resume: resume training
        cifar: type of cifar (10 or 100)
        val_split_size: number of elements in validation part of training data split
        val_split_existing: folder with existing val-train split
        output__ood: whether to compute and save outputs on ood dataset (the other cifar)

    Returns:

    """
    args = locals()
    args = namedtuple('args', args.keys())(*args.values())

    net = get_network(args)

    # data preprocessing:
    cifar_train_loader, cifar_val_loader = get_train_val_split_dataloader(val_count=val_split_size,
                                                                          existing_train_val_split=val_split_existing,
                                                                          cifar_type=cifar,
                                                                          num_workers=4,
                                                                          batch_size=b,
                                                                          shuffle=True,
                                                                          data_folder=None if cifar_data=="" else cifar_data)

    cifar_test_loader = get_test_dataloader_general(cifar_type=cifar, num_workers=4, batch_size=b, shuffle=False,
                                                                          data_folder=None if cifar_data=="" else cifar_data)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(cifar_train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.device != 'cpu':
        input_tensor = input_tensor.to(torch.device(args.device))
    writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False, net=net, loss_function=loss_function, test_loader=cifar_test_loader,
                            writer=writer, args=args)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch=epoch, net=net, warmup_scheduler=warmup_scheduler, training_loader=cifar_train_loader,
              optimizer=optimizer, loss_function=loss_function, writer=writer, args=args)
        acc = eval_training(epoch=epoch, net=net, loss_function=loss_function, test_loader=cifar_test_loader,
                            writer=writer, args=args)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
    produce_outputs(net, args)
