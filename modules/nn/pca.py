import numpy as np
import torch

class PCA:
    def _standardize(self,X):
        X_mean = torch.mean(X,dim=0)
        X_std = torch.std(X,dim=0)

        X_norm = (X - X_mean) / X_std

        return X_norm, X_mean, X_std
    
    def _perform_pca(self,X_norm):
        U, S, V = torch.svd(X_norm)
        
        X_pca = torch.matmul(X_norm,V)

        return X_pca, V