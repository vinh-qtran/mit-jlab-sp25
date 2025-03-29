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

class GNNLayer(nn.Module):
    def __init__(self,
                 input_dim : int,
                 hidden_node_dims : list,
                 hidden_message_dims : list,
                 hidden_feedforward_dims : list,
                 output_dim : int,
                 dropout = 0.0,
                 activation = nn.ReLU(),
                 norm = nn.BatchNorm1d,):
        super(GNNLayer, self).__init__()

        self.node = BaseMLP(input_dim=input_dim,
                           hidden_dims=hidden_node_dims[:-1],
                           output_dim=hidden_node_dims[-1],
                           dropout=dropout,
                           activation=activation,
                           norm=norm)
        self.message = BaseMLP(input_dim=input_dim*2,
                               hidden_dims=hidden_message_dims[:-1],
                               output_dim=hidden_message_dims[-1],
                               dropout=dropout,
                               activation=activation,
                               norm=norm)

        self.feedforward = BaseMLP(input_dim=hidden_node_dims[-1] + hidden_message_dims[-1],
                                   hidden_dims=hidden_feedforward_dims,
                                   output_dim=output_dim,
                                   dropout=dropout,
                                   activation=activation,
                                   norm=norm)
        
    def forward(self, x):
        x_node = self.node(x)
        
        messages = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    messages.append(torch.cat([x[:,i,:], x[:,j,:]], dim=1).view(x.shape[0], 1, -1))
        messages = torch.stack(messages, dim=1).view(x.shape[0], 4, 3, -1)
        x_messages = self.message(messages)

        x_aggregated = torch.mean(x_messages, dim=2)
        
        x_out = self.feedforward(torch.cat([x_node, x_aggregated], dim=-1))
        return x_out

class LeptonGNN(nn.Module):
    def __init__(self,
                 lepton_dim : int = 6,
                 pid_embedding_dim : int = 1,
                 hidden_lepton_dims : list = [32],
                 hidden_gnn_args : dict = {
                     'input_dim' : [16,16,16],
                     'hidden_node_dims' : [[32,8],[32,8],[32,8]],
                     'hidden_message_dims' : [[32,8],[32,8],[32,8]],
                     'hidden_feedforward_dims' : [[16,8],[16,8],[16,8]],
                 },
                 hidden_mpl_dims : list = [64, 16, 4],
                 output_dim : int = 1,
                 dropout : float = 0.0,
                 activation = nn.ReLU(),
                 norm = nn.BatchNorm1d,
                 last_activation = nn.Sigmoid()):
        super(LeptonGNN, self).__init__()

        self.include_pid = pid_embedding_dim > 0
        if self.include_pid:
            self.pid_embedding = nn.Embedding(4, pid_embedding_dim)

        self.lepton_layers = BaseMLP(input_dim=lepton_dim,
                                     hidden_dims=hidden_lepton_dims,
                                     output_dim=hidden_gnn_args['input_dim'][0], 
                                     dropout=dropout,
                                     activation=activation, 
                                     norm=norm, 
                                     last_activation=nn.ReLU())

        self.gnn_layers = nn.ModuleList()
        for i in range(len(hidden_gnn_args['input_dim'])):
            self.gnn_layers.append(GNNLayer(input_dim=hidden_gnn_args['input_dim'][i],
                                            hidden_node_dims=hidden_gnn_args['hidden_node_dims'][i],
                                            hidden_message_dims=hidden_gnn_args['hidden_message_dims'][i],
                                            hidden_feedforward_dims=hidden_gnn_args['hidden_feedforward_dims'][i],
                                            output_dim=hidden_gnn_args['input_dim'][i+1] if i != len(hidden_gnn_args['input_dim'])-1 else hidden_mpl_dims[0],
                                            dropout=dropout,
                                            activation=activation,
                                            norm=norm))
            
        self.mlp_layers = BaseMLP(input_dim=hidden_mpl_dims[0],
                                  hidden_dims=hidden_mpl_dims[1:],
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

        for gnn_layer in self.gnn_layers:
            x_leptons = gnn_layer(x_leptons)
        
        x_mlp = x_leptons.mean(dim=1)
        return self.mlp_layers(x_mlp)