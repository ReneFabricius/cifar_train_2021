import torch
from torch.utils.data import Dataset, DataLoader

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features_ = features
        self.labels_ = labels
        
    def __len__(self):
        return self.labels_.shape[0]
    
    def __getitem__(self, idx):
        return self.features_[idx], self.labels_[idx]
    

class TransferLearner:
    def __init__(self, C=1.0, fit_intercept=True, epochs=25, verbosity=0, device="cpu"):
        self.C_ = C
        self.fit_intercept_ = fit_intercept
        self.coefs_ = None
        self.epochs_ = epochs
        self.verbosity_ = verbosity
        self.dev_ = device
        
    def decision_function(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.fit_intercept_:
            dec = torch.mm(X, self.coefs_[:, :-1].T) + self.coefs_[:, -1]
        else:
            dec = torch.mm(X, self.coefs_.T) 
        return dec
    
    def fit(self, X, y, batch_size=1024):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            micro_batch (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        X.requires_grad_(False)
        y.requires_grad_(False)
        
        dtp = X.dtype
        
        classes = torch.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        
        coefs = torch.randn(size=(n_classes, n_features + int(self.fit_intercept_)), dtype=dtp, device=self.dev_, requires_grad=True)
        ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        opt = torch.optim.SGD(params=(coefs, ), lr=0.1, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=int(0.8 * self.epochs_ / 3.0), gamma=0.1)
        
        train_dataset = FeatureDataset(features=X, labels=y)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(self.epochs_):
            if self.verbosity_ > 0:
                print("Epoch {}/{}".format(epoch, self.epochs_ - 1))
                
            running_loss = 0.0
            running_corrects = 0
            
            for feat, lab in train_dataloader:
                feat = feat.to(self.dev_)
                lab = lab.to(self.dev_)
                
                opt.zero_grad()
                if self.fit_intercept_:
                    Ws = coefs[:, :-1]
                    Bs = coefs[:, -1]
                else:
                    Ws = coefs
                    
                with torch.set_grad_enabled(True):
                    lin_comb = torch.mm(feat, Ws.T)
                    if self.fit_intercept_:
                        lin_comb += Bs
                
                _, preds = torch.max(lin_comb, dim=1)
                with torch.set_grad_enabled(True):
                    loss = ce_loss(lin_comb, lab)
                
                running_loss += loss.item() * feat.size(0)
                running_corrects += torch.sum(preds == lab.data)
                
                with torch.set_grad_enabled(True):
                    L2 = torch.sum(torch.pow(Ws, 2))
                    loss += L2 / (n_features * n_classes * self.C_)

                loss.backward()
                
                print("Grad min: {}, max: {}".format(torch.min(coefs.grad), torch.max(coefs.grad)))
                
                opt.step()
            
            lr_scheduler.step()
            
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset)
            
            if self.verbosity_ > 0:
                print("Loss: {:.4f}, Acc: {:.4f}".format(epoch_loss, epoch_acc))
                

        coefs.requires_grad_(False)
        if not self.fit_intercept_:
            zero_interc = torch.zeros(size=(n_classes, 1), dtype=dtp, device=dev)
            coefs = torch.cat([coefs, zero_interc], dim=-1)
        
        self.coefs_ = coefs
                
        
    def predict_proba(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        sm = torch.nn.Softmax(dim=1)
        dec = self.decision_function(X)
        return sm(dec)    
    