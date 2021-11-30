import os
import clip
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import re
import argparse
import numpy as np
import shutil

from torchvision import datasets
from conf import settings
from utils import get_test_dataloader_general, get_train_val_split_dataloader, get_cifar_labels

@torch.no_grad()
def infer_clip():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-cifar_data', type=str, help='Folder where cifar dataset should be stored')
    parser.add_argument('-clip_data', type=str, help='Folder where clip model should be stored')
    parser.add_argument('-batch_sz', type=int, default=64, help='batch size')
    parser.add_argument('-device', type=str, default="cpu", help='Device on which to perform the computations')
    parser.add_argument('-cifar', default=100, type=int, help='cifar type (10 or 100)')
    parser.add_argument('-architecture', type=str, default='ViT-B/32', help='Clip architecture')
    args = parser.parse_args()
    
    fold_ptrn = r'^\d+$'
    folders = [f for f in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, f))]
    repl_folders = [f for f in folders if re.match(fold_ptrn, f) is not None]
    print("Replications found {}".format(repl_folders))

    print("Loading clip model")
    model, preprocess = clip.load(args.architecture, device=args.device, download_root=args.clip_data)
    model.eval().float()
    if args.cifar == 10:
        cifar_train = torchvision.datasets.CIFAR10(root=args.cifar_data, train=True, download=True, transform=preprocess)
        cifar_test = torchvision.datasets.CIFAR10(root=args.cifar_data, train=False, download=True, transform=preprocess)
    elif args.cifar == 100:
        cifar_train = torchvision.datasets.CIFAR100(root=args.cifar_data, train=True, download=True, transform=preprocess)
        cifar_test = torchvision.datasets.CIFAR100(root=args.cifar_data, train=False, download=True, transform=preprocess)
    else:
        print("Error: unsupported cifar type {}".format(args.cifar))
        return 1
    
    train_loader = DataLoader(dataset=cifar_train, batch_size=args.batch_sz, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=cifar_test, batch_size=args.batch_sz, shuffle=False, num_workers=4)

    labels = cifar_test.classes
    text_inputs = torch.cat([clip.tokenize("a photo of a {}".format(c)) for c in labels]).to(args.device)
    text_features = model.encode_text(text_inputs)
    
    train_features = []
    test_features = []
    
    print("Processing train data")
    for batch_index, (images, labels) in enumerate(train_loader):
        print("Progress {}%".format(100 * (batch_index + 1) // len(train_loader)), end="\r")
        with torch.no_grad():
            train_features.append(model.encode_image(images.to(args.device)))
    print("\n") 
    print("Processing test data")
    for batch_index, (images, labels) in enumerate(test_loader):
        print("Progress {}%".format(100 * (batch_index + 1) // len(test_loader)), end="\r")
        with torch.no_grad():
            test_features.append(model.encode_image(images.to(args.device)))
    print("\n") 

    train_features = torch.cat(train_features, dim=0)
    test_features = torch.cat(test_features, dim=0)
    
    text_features /= torch.linalg.vector_norm(text_features, dim=-1, keepdim=True)
    train_features /= torch.linalg.vector_norm(train_features, dim=-1, keepdim=True)
    test_features /= torch.linalg.vector_norm(test_features, dim=-1, keepdim=True)
    train_logits = 100.0 * train_features @ text_features.T
    test_logits = 100.0 * test_features @ text_features.T 
    
    for repl_f in repl_folders:
        print("Processing replication {}".format(repl_f))
        train_idx = torch.from_numpy(np.load(os.path.join(args.folder, repl_f, "split", "train_idx.npy"))).to(args.device)
        val_idx = torch.from_numpy(np.load(os.path.join(args.folder, repl_f, "split", "val_idx.npy"))).to(args.device)

        net_folder = os.path.join(args.folder, repl_f, "outputs", "clip_{}".format(args.architecture.replace('/', '-')))
        
        if not os.path.exists(net_folder): 
            os.mkdir(net_folder)
        
        np.save(os.path.join(net_folder, "train_outputs.npy"), train_logits[train_idx].cpu())
        np.save(os.path.join(net_folder, "train_labels.npy"), torch.tensor(cifar_train.targets)[train_idx].cpu())
        if len(val_idx) > 0:
            np.save(os.path.join(net_folder, "val_outputs.npy"), train_logits[val_idx].cpu())
            np.save(os.path.join(net_folder, "val_labels.npy"), torch.tensor(cifar_train.targets)[val_idx].cpu())
        np.save(os.path.join(net_folder, "test_outputs.npy"), test_logits.cpu())
        np.save(os.path.join(net_folder, "test_labels.npy"), torch.tensor(cifar_test.targets).cpu())
        

if __name__ == "__main__":
    infer_clip()