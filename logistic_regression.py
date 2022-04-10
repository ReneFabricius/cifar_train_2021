import torch

class LogisticRegressionTorch:
    """_summary_
    """
    
    def __init__(self, C=1.0, fit_intercept=True, max_iter=100):
        """_summary_

        Args:
            C (float, optional): _description_. Defaults to 1.0.
            fit_intercept (bool, optional): _description_. Defaults to True.
            max_iter (int, optional): _description_. Defaults to 100.
        """
        self.C_ = C
        self.fit_intercept_ = fit_intercept
        self.max_iter_ = max_iter
        self.coefs_ = None
        
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
    
    def fit(self, X, y, micro_batch=None):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            micro_batch (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        X.requires_grad(False)
        y.requires_grad(False)
        
        dev = X.device
        dtp = X.dtype
        
        classes = torch.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        
        coefs = torch.zeros(size=(n_classes, n_features + int(self.fit_intercept_)), dtype=dtp, device=dev, requires_grad=True)
        ce_loss = torch.nn.CrossEntropyLoss(reduction="sum")
        opt = torch.optim.LBFGS(params=(coefs,), max_iter=self.max_iter_)
        
        if micro_batch is None:
            micro_batch = X.shape[0]
            
        def closure_loss():
            opt.zero_grad()
            if self.fit_intercept_:
                Ws = coefs[:, :-1]
                Bs = coefs[:, -1]
            else:
                Ws = coefs
            
            loss = torch.tensor([0], device=dev, dtype=dtp)
            for mbs in range(0, n_samples, micro_batch):
                cur_X = X[mbs : mbs + micro_batch]
                cur_y = y[mbs : mbs + micro_batch]
                lin_comb = torch.mm(cur_X, Ws.T)
                if self.fit_intercept_:
                    lin_comb += Bs
                
                loss += ce_loss(lin_comb, cur_y)
                
            loss /= n_samples
            
            L2 = torch.sum(torch.pow(Ws, 2))
            loss += L2 / (n_features * n_classes * self.C_)
            
            loss.backward(retain_graph=True)
            return loss
            
        opt.step(closure_loss)
        
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