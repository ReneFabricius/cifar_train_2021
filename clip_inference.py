import os
import clip
from numpy.core.function_base import linspace
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import re
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import shutil

from torchvision import datasets
from conf import settings

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
    parser.add_argument('-verbosity', default=0, type=int, help='Verbosity level')
    parser.add_argument('-linear_probe', dest='lin_probe', action='store_true', help='Whether to use linear probe to infer predictions')
    parser.set_defaults(lin_probe=False)
    args = parser.parse_args()
    
    fold_ptrn = r'^\d+$'
    folders = [f for f in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, f))]
    repl_folders = [f for f in folders if re.match(fold_ptrn, f) is not None]
    if args.verbosity > 0:
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

    if not args.lin_probe:
        print("Processing text data")
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
    train_features /= torch.linalg.vector_norm(train_features, dim=-1, keepdim=True)
    test_features /= torch.linalg.vector_norm(test_features, dim=-1, keepdim=True)
        
    if not args.lin_probe:
        text_features /= torch.linalg.vector_norm(text_features, dim=-1, keepdim=True)
        train_logits = 100.0 * train_features @ text_features.T
        test_logits = 100.0 * test_features @ text_features.T 
        
    for repl_f in repl_folders:
        print("Processing replication {}".format(repl_f))
        train_idx = torch.from_numpy(np.load(os.path.join(args.folder, repl_f, "split", "train_idx.npy"))).to(args.device)
        val_idx = torch.from_numpy(np.load(os.path.join(args.folder, repl_f, "split", "val_idx.npy"))).to(args.device)
        
        if args.lin_probe:
            lin_val_set_size = 5000
            C_start = 0.1
            C_end = 2
            C_step = 0.1
            C_vals = np.linspace(start=C_start, stop=C_end,
                              num=int((C_end - C_start) // C_step + 2), endpoint=True)
            if len(val_idx) == 0:
                lin_train_idx, lin_val_idx = train_test_split(np.arange(train_features.shape[0]), test_size=lin_val_set_size,
                                                              shuffle=True, stratify=cifar_train.targets)
                lin_train_idx = torch.from_numpy(lin_train_idx).to(device=args.device, dtype=torch.long)
                lin_val_idx = torch.from_numpy(lin_val_idx).to(device=args.device, dtype=torch.long)
            else:
                lin_train_idx = train_idx
                lin_val_idx = val_idx
            
            train_targets = torch.tensor(cifar_train.targets, device=args.device)
            
            lin_train_features = train_features[lin_train_idx]
            lin_val_features = train_features[lin_val_idx]
            lin_train_tar = train_targets[lin_train_idx]
            lin_val_tar = train_targets[lin_val_idx]                

            best_acc = 0
            best_C = 1.0
            best_model = None
            for C_val in C_vals:
                if args.verbosity > 0:
                    print("Testing C value {}".format(C_val))
                    
                log_reg = LogisticRegression(solver="sag", penalty='l2', max_iter=1000, C=C_val, verbose=args.verbosity, multi_class="multinomial")
                log_reg.fit(X=lin_train_features.cpu(), y=lin_train_tar.cpu())            
                val_pred = torch.tensor(log_reg.decision_function(lin_val_features.cpu()), device=args.device)
                cur_acc = torch.sum(val_pred.topk(k=1, dim=-1).indices.squeeze() == lin_val_tar).item() / len(lin_val_tar)
                if args.verbosity > 0:
                    print("Validation accuracy obtained {}".format(cur_acc))
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    best_C = C_val
                    best_model = log_reg

            print("C value selected {} with validation accuracy {}".format(best_C, best_acc))
            train_logits = torch.tensor(best_model.decision_function(train_features.cpu()), device=args.device)
            test_logits = torch.tensor(best_model.decision_function(test_features.cpu()), device=args.device)

        clip_name = "clip_{}".format(args.architecture.replace('/', '-')) + ("_LP" if args.lin_probe else "")
        net_folder = os.path.join(args.folder, repl_f, "outputs", clip_name) 
        
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