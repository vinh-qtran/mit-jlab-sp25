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
    
    def transform(self, X):
        """
        Transform data using the fitted PCA model
        
        Parameters:
        X - input data (numpy array or torch tensor)
        
        Returns:
        X_pca - PCA transformed data
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Standardize using stored mean and std
        X_norm = (X - self.X_mean) / self.X_std
        
        # Apply PCA transformation using stored rotation matrix
        X_pca = torch.matmul(X_norm, self.V)
        
        # Convert back to numpy for easier integration with other libraries
        return X_pca.numpy() if torch.is_tensor(X_pca) else X_pca

    # Add fit method to your PCA class
    def fit(self, X):
        """
        Fit PCA model to the data
        
        Parameters:
        X - input data (numpy array or torch tensor)
        
        Returns:
        self - fitted PCA model
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Standardize data
        self.X_norm, self.X_mean, self.X_std = self._standardize(X)
        
        # Perform PCA
        self.X_pca, self.V = self._perform_pca(self.X_norm)
        
        return self

    # Add fit_transform method for convenience
    def fit_transform(self, X):
        """
        Fit PCA model and transform the data
        
        Parameters:
        X - input data (numpy array or torch tensor)
        
        Returns:
        X_pca - PCA transformed data
        """
        self.fit(X)
        return self.X_pca.numpy() if torch.is_tensor(self.X_pca) else self.X_pca