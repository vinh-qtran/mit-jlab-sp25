import os
import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

##########################################################################################

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pickle

import params, utils
from models import mlp, gnn
from modules import training

import multiprocessing as mp
mp.set_start_method('fork')

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

##########################################################################################

import importlib

importlib.reload(params)
importlib.reload(utils)

importlib.reload(training)

importlib.reload(mlp)
importlib.reload(gnn)

##########################################################################################

MC_dir = '../HiggsTo4L/MC/'
data_dir = '../HiggsTo4L/data/'

four_leptons_data = utils.FourLeptonsData()
four_leptons_nn = utils.FourLeptonNN()

##########################################################################################

all_higgs_and_zz_data = [
    four_leptons_data.read_data(file) for file in [
        MC_dir + 'higgs2011.csv',
        MC_dir + 'higgs2012.csv',

        MC_dir + 'zzto4mu2011.csv',
        MC_dir + 'zzto2mu2e2011.csv',
        MC_dir + 'zzto4e2011.csv',
        MC_dir + 'zzto4mu2012.csv',
        MC_dir + 'zzto2mu2e2012.csv',
        MC_dir + 'zzto4e2012.csv',
    ]
]

all_higgs_and_zz_data = four_leptons_data.apply_basic_cuts(
    all_higgs_and_zz_data,
    # inv_mass_cut=None,
    show_cut_info=False
)

fields = [
    'PID1', 'px1', 'py1',
    'PID2', 'px2', 'py2',
    'PID3', 'px3', 'py3',
    'PID4', 'px4', 'py4',
]

test_ratio = 0.1

higgs_and_zz_train_loader, higgs_and_zz_val_loader, higgs_and_zz_test_datasets, higgs_and_zz_test_indices, N_trainval, N_test = four_leptons_nn.get_dataloaders(
    all_data = all_higgs_and_zz_data, data_labels = [1]*2 + [0]*6, fields = fields,
    test_ratio = test_ratio, train_ratio=0.8,
    balance_categories = True,
    batch_size=2048,num_workers=8,
    seed=42,
)

print('Training dataset size:', N_trainval)
print('Test dataset size:', N_test)

##########################################################################################

model = mlp.LeptonMLP(
    lepton_dim = 3,
    pid_embedding_dim = 1,
    hidden_lepton_dims = [16, 16, 16],
    hidden_mpl_dims = [32, 16, 8],
    output_dim = 1,
    dropout = 0.0,
    activation = nn.ReLU(),
    norm = nn.BatchNorm1d,
    last_activation = nn.Sigmoid(),
)

# model = gnn.LeptonGNN(
#     lepton_dim = 3,
#     pid_embedding_dim = 1,
#     hidden_lepton_dims = [128,64,],
#     hidden_gnn_args = {
#         'input_dim' : [32,32,32],
#         'hidden_node_dims' : [[64,16],[64,16],[64,16]],
#         'hidden_message_dims' : [[64,16],[64,16],[64,16]],
#         'hidden_feedforward_dims' : [[64,32],[64,32],[64,32]],
#     },
#     hidden_mpl_dims = [64, 16, 4],
#     output_dim = 1,
#     dropout = 0.0,
#     activation = nn.ReLU(),
#     norm = nn.BatchNorm1d,
#     last_activation = nn.Sigmoid(),
# )

##########################################################################################

trainer = training.SupervisedTraining(
    model=model,
    train_loader=higgs_and_zz_train_loader,
    val_loader=higgs_and_zz_val_loader,
    num_epochs=100,
    lr=2e-4,
    criterion=nn.BCELoss(),
    optimizer=optim.Adam,
    scheduler=optim.lr_scheduler.CosineAnnealingLR,
    scheduler_params={'T_max': 50},
    is_classification=True,
    num_classes=2,
    device='mps',
)

##########################################################################################

trainer.train(save_training_stats_every=5, save_model_every=None, outpath='training_result/lepton_mlp_limited_mass_pT_and_ID/')