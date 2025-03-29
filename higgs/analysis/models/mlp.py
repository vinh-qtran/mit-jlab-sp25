import numpy as np

import torch
import torch.nn as nn

class BaseMLP(nn.Module):
    def __init__(self,
                 input_dim : int,
                 hidden_dims : list,
                 output_dim : int = 1,
                 dropout = 0.0,
                 activation = nn.ReLU(),
                 norm = nn.BatchNorm1d,
                 last_activation = nn.Sigmoid()):
        super(BaseMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims)-1):
            self.norms.append(norm(layer_dims[i]))

            self.layers.append(nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                activation if i != len(layer_dims)-2 else last_activation,
            ))

    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = norm(x.view(-1, x.shape[-1])).view(x.shape)
            x = layer(x)
        return x
    
class LeptonMLP(nn.Module):
    def __init__(self,
                 lepton_dim : int = 6,
                 pid_embedding_dim : int = 1,
                 hidden_lepton_dims : list = [64, 64, 64],
                 hidden_mpl_dims : list = [64, 16, 4],
                 output_dim : int = 1,
                 dropout : float = 0.0,
                 activation = nn.ReLU(),
                 norm = nn.BatchNorm1d,
                 last_activation = nn.Sigmoid()):
        super(LeptonMLP, self).__init__()

        self.include_pid = pid_embedding_dim > 0
        if self.include_pid:
            self.pid_embedding = nn.Embedding(4, pid_embedding_dim)

        self.lepton_layers = BaseMLP(input_dim=lepton_dim,
                                     hidden_dims=hidden_lepton_dims[:-1],
                                     output_dim=hidden_lepton_dims[-1], 
                                     dropout=dropout, 
                                     activation=activation, 
                                     norm=norm, 
                                     last_activation=nn.ReLU())
        
        self.mlp_layers = BaseMLP(input_dim=hidden_lepton_dims[-1],
                                  hidden_dims=hidden_mpl_dims,
                                  output_dim=output_dim,
                                  dropout=dropout,
                                  activation=activation,
                                  norm=norm,
                                  last_activation=last_activation)

    def forward(self, x):
        x_leptons = x.view(x.shape[0], 4, -1)

        if self.include_pid:
            x_pid = self.pid_embedding(x_leptons[:, :, 0].long())
            x_leptons = torch.cat([x_pid, x_leptons[:, :, 1:]], dim=-1)

        x_leptons = self.lepton_layers(x_leptons)
        x_mlp = x_leptons.mean(dim=1)
        return self.mlp_layers(x_mlp)