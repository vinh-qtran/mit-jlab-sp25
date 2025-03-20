import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

import utils
import importlib
importlib.reload(utils)

class FourLeptonsDataset:
    def __init__(self,
                 dfs : list, label_sets : list,
                 fields : list,
                 balance_classes : bool = True,
                 train_ratio : float = 0.8,
                 batch_size : int = 2048,
                 num_workers : int = 8,
                 seed : int = 42,):
        self.fields = fields
        self.balance_classes = balance_classes

        self.X, self.Y = self._get_data(dfs,label_sets)

        self.train_ratio = train_ratio

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.seed = seed
        torch.manual_seed(self.seed)

        self.train_loader, self.val_loader, self.N_data, self.N_train, self.N_val = self._get_dataloaders()

    def _get_data(self,dfs,label_sets):
        if len(dfs) != len(label_sets):
            raise ValueError('Number of dataframes and label_sets must be the same')
        
        data = pd.concat(dfs,ignore_index=True)[self.fields].to_numpy()
        labels = np.concatenate(label_sets,axis=0)

        if self.balance_classes:
            class_0_indices = np.where(labels == 0)[0]
            class_1_indices = np.where(labels == 1)[0]

            N_subsample = min(len(class_0_indices),len(class_1_indices))

            reduced_class_0_indices = np.random.choice(class_0_indices,N_subsample,replace=False)
            reduced_class_1_indices = np.random.choice(class_1_indices,N_subsample,replace=False)

            indices = np.concatenate([reduced_class_0_indices,reduced_class_1_indices],axis=0)

            data = data[indices]
            labels = labels[indices]

        X = torch.tensor(data,dtype=torch.float32)
        Y = torch.tensor(labels,dtype=torch.float32)

        return X,Y
    
    def _get_dataloaders(self):
        dataset = TensorDataset(self.X,self.Y)

        N_data = dataset.tensors[0].shape[0]
        N_train = int(self.train_ratio*N_data)
        N_val = N_data - N_train

        train_dataset, val_dataset = torch.utils.data.random_split(dataset,[N_train,N_val])

        train_loader = DataLoader(train_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=True,)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,)
        
        return train_loader, val_loader, N_data, N_train, N_val