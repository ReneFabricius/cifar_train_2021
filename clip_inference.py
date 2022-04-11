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
from logistic_regression import LogisticRegressionTorch
from transfer_learner import TransferLearner
import numpy as np
from timeit import default_timer as timer
from weensembles.utils import cuda_mem_try

from torchvision import datasets
from conf import settings

@torch.no_grad()
def infer_clip():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', type=str, required=True, help='experiment root folder')
    parser.add_argument('-dataset_data', type=str, help='Folder where dataset is stored')
    parser.add_argument('-dataset', default="cifar10", choices=["cifar10", "cifar100", "imagenet"], help="Dataset to use.")
    parser.add_argument('-clip_data', type=str, help='Folder where clip model should be stored')
    parser.add_argument('-batch_sz', type=int, default=64, help='batch size')
    parser.add_argument('-device', type=str, default="cpu", help='Device on which to perform the computations')
    parser.add_argument('-architecture', type=str, default='ViT-B/32', help='Clip architecture')
    parser.add_argument('-verbosity', default=0, type=int, help='Verbosity level')
    parser.add_argument('-multiple_repl', action="store_true", dest="multi_repl", help="Root folder contains multiple replications folders")
    parser.add_argument('-load_features', action="store_true", dest="load_feat", help="If specified, training and testing features are loaded. Expected location is the architecture subfolder of the root folder.")
    args = parser.parse_args()
    
    clip_name = "clip_{}".format(args.architecture.replace('/', '-')) + "_LP"

    if args.multi_repl:
        fold_ptrn = r'^\d+$'
        folders = [f for f in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, f))]
        repl_folders = sorted([f for f in folders if re.match(fold_ptrn, f) is not None])
        if args.verbosity > 0:
            print("Replications found {}".format(repl_folders))
    else:
        repl_folders = [""]
        
    has_saved_features = os.path.exists(
        os.path.join(args.folder, clip_name, "train_features.npy")
        ) and os.path.exists(
            os.path.join(args.folder, clip_name, "test_features.npy"))
        
    has_saved_targets = os.path.exists(
        os.path.join(args.folder, clip_name, "train_targets.npy")
        ) and os.path.exists(
            os.path.join(args.folder, clip_name, "test_targets.npy"))

    if (not args.load_feat) or (not has_saved_features):
        print("Loading clip model")
        model, preprocess = clip.load(args.architecture, device=args.device, download_root=args.clip_data)
        model.eval().float()
    else:
        preprocess = None
    
    if (not args.load_feat) or (not has_saved_features) or (not has_saved_targets):
        print("Loading dataset")
        if args.dataset == "cifar10":
            dataset_train = torchvision.datasets.CIFAR10(root=args.dataset_data, train=True, download=True, transform=preprocess)
            dataset_test = torchvision.datasets.CIFAR10(root=args.dataset_data, train=False, download=True, transform=preprocess)
        elif args.dataset == "cifar100":
            dataset_train = torchvision.datasets.CIFAR100(root=args.dataset_data, train=True, download=True, transform=preprocess)
            dataset_test = torchvision.datasets.CIFAR100(root=args.dataset_data, train=False, download=True, transform=preprocess)
        elif args.dataset == "imagenet":
            traindir = os.path.join(args.dataset_data, "train")
            testdir = os.path.join(args.dataset_data, "val")
            dataset_train = torchvision.datasets.ImageFolder(root=traindir, transform=preprocess)
            dataset_test = torchvision.datasets.ImageFolder(root=testdir, transform=preprocess)
        else:
            print("Error: unsupported dataset type {}".format(args.dataset))
            return 1
    
    if (not args.load_feat) or (not has_saved_features):
        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_sz, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_sz, shuffle=False, num_workers=4, pin_memory=True)
        
        train_features = []
        test_features = []
        
        print("Processing train data")
        for batch_index, (images, labels) in enumerate(train_loader):
            print("Progress {}%".format(100 * (batch_index + 1) // len(train_loader)), end="\r")
            with torch.no_grad():
                train_features.append(model.encode_image(images.to(args.device)).cpu())
        print("\n") 
        print("Processing test data")
        for batch_index, (images, labels) in enumerate(test_loader):
            print("Progress {}%".format(100 * (batch_index + 1) // len(test_loader)), end="\r")
            with torch.no_grad():
                test_features.append(model.encode_image(images.to(args.device)).cpu())
        print("\n") 

        train_features = torch.cat(train_features, dim=0).to(args.device)
        test_features = torch.cat(test_features, dim=0).to(args.device)

        if not os.path.exists(os.path.join(args.folder, clip_name)):
            os.mkdir(os.path.join(args.folder, clip_name))

        print("Saving features")
        np.save(os.path.join(args.folder, clip_name, "train_features.npy"), train_features.cpu())
        np.save(os.path.join(args.folder, clip_name, "test_features.npy"), test_features.cpu())
        
    if (not args.load_feat) or (not has_saved_targets):
        if not os.path.exists(os.path.join(args.folder, clip_name)):
            os.mkdir(os.path.join(args.folder, clip_name))
        print("Saving targets")
        train_targets = np.array(dataset_train.targets)
        test_targets = np.array(dataset_test.targets)
        np.save(os.path.join(args.folder, clip_name, "train_targets.npy"), train_targets)
        np.save(os.path.join(args.folder, clip_name, "test_targets.npy"), test_targets)

    if args.load_feat and has_saved_features:
        print("Loading features")
        train_features = torch.from_numpy(np.load(os.path.join(args.folder, clip_name, "train_features.npy"))).to(args.device)
        test_features = torch.from_numpy(np.load(os.path.join(args.folder, clip_name, "test_features.npy"))).to(args.device)
        
    if args.load_feat and has_saved_targets:
        train_targets = np.load(os.path.join(args.folder, clip_name, "train_targets.npy"))
        test_targets = np.load(os.path.join(args.folder, clip_name, "test_targets.npy"))
        
    train_features /= torch.linalg.vector_norm(train_features, dim=-1, keepdim=True)
    test_features /= torch.linalg.vector_norm(test_features, dim=-1, keepdim=True)
    train_targets = torch.from_numpy(train_targets).to(args.device)
    test_targets = torch.from_numpy(test_targets).to(args.device)
        
    for repl_f in repl_folders:
        print("Processing subfolder {}".format(repl_f))
        train_idx = torch.from_numpy(np.load(os.path.join(args.folder, repl_f, "split", "train_idx.npy"))).to(args.device)
        val_idx = torch.from_numpy(np.load(os.path.join(args.folder, repl_f, "split", "val_idx.npy"))).to(args.device)
        
        start = timer()
        lin_val_set_size = 5000
        E_start = -1.7
        E_end = 1.7
        E_count = 21
        C_vals = 10**np.linspace(start=E_start, stop=E_end,
                            num=E_count, endpoint=True)
        if len(val_idx) == 0:
            lin_train_idx, lin_val_idx = train_test_split(np.arange(train_features.shape[0]), test_size=lin_val_set_size,
                                                            shuffle=True, stratify=train_targets)
            lin_train_idx = torch.from_numpy(lin_train_idx).to(device=args.device, dtype=torch.long)
            lin_val_idx = torch.from_numpy(lin_val_idx).to(device=args.device, dtype=torch.long)
        else:
            lin_train_idx = train_idx
            lin_val_idx = val_idx
        
        lin_train_features = train_features[lin_train_idx]
        lin_val_features = train_features[lin_val_idx]
        lin_train_tar = train_targets[lin_train_idx]
        lin_val_tar = train_targets[lin_val_idx]                

        best_acc = 0
        best_C = 1.0
        best_model = None
        for Ci, C_val in enumerate(C_vals):
            if args.verbosity == 0:
                print("Progress {}%".format(100 * (Ci + 1) // len(C_vals)), end="\r")
            if args.verbosity > 0:
                print("Testing C value {}".format(C_val))
                
            #log_reg = LogisticRegression(solver="sag", penalty='l2', max_iter=1000, C=C_val, verbose=args.verbosity, multi_class="multinomial")
            #log_reg = LogisticRegressionTorch(C=C_val, fit_intercept=True, max_iter=100)
            transf_lear = TransferLearner(C=C_val, fit_intercept=True, epochs=25, verbosity=args.verbosity)
            cuda_mem_try(
                fun=lambda batch_size: transf_lear.fit(X=lin_train_features, y=lin_train_tar, batch_size=batch_size),
                start_bsz=4096,
                device=args.device,
                dec_coef=0.8,
                verbose=args.verbosity)
            
            val_pred = transf_lear.decision_function(lin_val_features)
            cur_acc = torch.sum(val_pred.topk(k=1, dim=-1).indices.squeeze() == lin_val_tar).item() / len(lin_val_tar)
            if args.verbosity > 0:
                print("Validation accuracy obtained {}".format(cur_acc))
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_C = C_val
                best_model = transf_lear

            print("C value selected {} with validation accuracy {}".format(best_C, best_acc))
            train_logits = best_model.decision_function(train_features)
            test_logits = best_model.decision_function(test_features)
            print("Linear probe inference finished in {}s".format(timer() - start))

        net_folder = os.path.join(args.folder, repl_f, "outputs", clip_name) 
        
        if not os.path.exists(net_folder): 
            os.mkdir(net_folder)
        
        np.save(os.path.join(net_folder, "train_outputs.npy"), train_logits[train_idx].cpu())
        np.save(os.path.join(net_folder, "train_labels.npy"), torch.tensor(train_targets)[train_idx].cpu())
        if len(val_idx) > 0:
            np.save(os.path.join(net_folder, "val_outputs.npy"), train_logits[val_idx].cpu())
            np.save(os.path.join(net_folder, "val_labels.npy"), torch.tensor(train_targets)[val_idx].cpu())
        np.save(os.path.join(net_folder, "test_outputs.npy"), test_logits.cpu())
        np.save(os.path.join(net_folder, "test_labels.npy"), torch.tensor(test_targets).cpu())
        

if __name__ == "__main__":
    infer_clip()