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
from torch.utils.data import TensorDataset, random_split, DataLoader

import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

##########################################################################################

class FourLeptonsData:
    field_indices = {
        'PID1' : 0, 'E1' : 1, 'px1' : 2, 'py1' : 3, 'pz1' : 4, 'eta1' : 5, 'cos_phi1' : 6, 'sin_phi1' : 7,
        'PID2' : 8, 'E2' : 9, 'px2' : 10, 'py2' : 11, 'pz2' : 12, 'eta2' : 13, 'cos_phi2' : 14, 'sin_phi2' : 15,
        'PID3' : 16, 'E3' : 17, 'px3' : 18, 'py3' : 19, 'pz3' : 20, 'eta3' : 21, 'cos_phi3' : 22, 'sin_phi3' : 23,
        'PID4' : 24, 'E4' : 25, 'px4' : 26, 'py4' : 27, 'pz4' : 28, 'eta4' : 29, 'cos_phi4' : 30, 'sin_phi4' : 31,
    }

    def _read_data(self,data_file):
        df = pd.read_csv(data_file)

        for i in range(1,5):
            df[f'cos_phi{i}'] = np.cos(df[f'phi{i}'])
            df[f'sin_phi{i}'] = np.sin(df[f'phi{i}'])

        data = df[[field for field in self.field_indices.keys()]].to_numpy()

        return data

    def get_pT(self,data,id):
        return np.sqrt(
            data[:,self.field_indices[f'px{id}']]**2 + data[:,self.field_indices[f'py{id}']]**2
        )

    def get_m4l(self,data):
        return np.sqrt(
            (data[:,self.field_indices['E1']] + data[:,self.field_indices['E2']] + data[:,self.field_indices['E3']] + data[:,self.field_indices['E4']])**2 - \
            (data[:,self.field_indices['px1']] + data[:,self.field_indices['px2']] + data[:,self.field_indices['px3']] + data[:,self.field_indices['px4']])**2 - \
            (data[:,self.field_indices['py1']] + data[:,self.field_indices['py2']] + data[:,self.field_indices['py3']] + data[:,self.field_indices['py4']])**2 - \
            (data[:,self.field_indices['pz1']] + data[:,self.field_indices['pz2']] + data[:,self.field_indices['pz3']] + data[:,self.field_indices['pz4']])**2
        )
    
    def _conservation_cut(self,data,show_cut_info=True):
        mask = np.sum(data[:,[self.field_indices['PID1'],self.field_indices['PID2'],self.field_indices['PID3'],self.field_indices['PID4']]],axis=1) == 0

        if show_cut_info:
            print(f' Conservation cut: {mask.sum()} events passed out of {len(mask)} ({100*mask.sum()/len(mask):.0f}%)')

        return data[mask]
    
    def _leptons_cut(self,data,lepton_pT_cuts=[7,5],lepton_eta_cuts=[2.5,2.4],show_cut_info=True):
        masks = [
            np.logical_or(
                np.logical_and(
                    np.abs(data[:,self.field_indices[f'PID{i+1}']]) == 11,
                    np.logical_and(self.get_pT(data,i+1) > lepton_pT_cuts[0], np.abs(data[:,self.field_indices[f'eta{i+1}']]) < lepton_eta_cuts[0])
                ),
                np.logical_and(
                    np.abs(data[:,self.field_indices[f'PID{i+1}']]) == 13,
                    np.logical_and(self.get_pT(data,i+1) > lepton_pT_cuts[1], np.abs(data[:,self.field_indices[f'eta{i+1}']]) < lepton_eta_cuts[1])
                )
            ).reshape((-1,1)) for i in range(4)
        ]

        mask = np.concatenate(masks,axis=1).all(axis=1)

        if show_cut_info:
            print(f' Leptons cut: {mask.sum()} events passed out of {len(mask)} ({100*mask.sum()/len(mask):.0f}%)')

        return data[mask]
    
    def _Z_mass_cut(self,data,heavier_Z_cuts=[40,120],lighter_Z_cuts=[12,120],show_cut_info=True):
        def _get_paired_Z_mass(paired_ids):
            return np.sqrt(
                (data[:,self.field_indices[f'E{paired_ids[0]}']] + data[:,self.field_indices[f'E{paired_ids[1]}']])**2 - \
                (data[:,self.field_indices[f'px{paired_ids[0]}']] + data[:,self.field_indices[f'px{paired_ids[1]}']])**2 - \
                (data[:,self.field_indices[f'py{paired_ids[0]}']] + data[:,self.field_indices[f'py{paired_ids[1]}']])**2 - \
                (data[:,self.field_indices[f'pz{paired_ids[0]}']] + data[:,self.field_indices[f'pz{paired_ids[1]}']])**2
            )

        masks = []
        for i in range(2,5):
            mZ1 = _get_paired_Z_mass([1,i])
            mZ2 = _get_paired_Z_mass([j for j in range(2,5) if j != i])

            good_pair_mask = data[:,self.field_indices['PID1']] + data[:,self.field_indices[f'PID{i}']] == 0

            heavier_Z1_mask = np.logical_and(
                np.logical_and(mZ1 > heavier_Z_cuts[0], mZ1 < heavier_Z_cuts[1]),
                np.logical_and(mZ2 > lighter_Z_cuts[0], mZ2 < lighter_Z_cuts[1])
            )

            lighter_Z1_mask = np.logical_and(
                np.logical_and(mZ1 > lighter_Z_cuts[0], mZ1 < lighter_Z_cuts[1]),
                np.logical_and(mZ2 > heavier_Z_cuts[0], mZ2 < heavier_Z_cuts[1])
            )

            masks.append(np.logical_and(good_pair_mask,np.logical_or(heavier_Z1_mask,lighter_Z1_mask)).reshape((-1,1)))

        mask = np.concatenate(masks,axis=1).any(axis=1)

        if show_cut_info:
            print(f' Z cut: {mask.sum()} events passed out of {len(mask)} ({100*mask.sum()/len(mask):.0f}%)')

        return data[mask]
    
    def apply_basic_cuts(self,all_data,
                         lepton_pT_cuts=[7,5],lepton_eta_cuts=[2.5,2.4],
                         heavier_Z_cuts=[40,120],lighter_Z_cuts=[12,120],
                         show_cut_info=True):
        reduced_data = []

        for data in all_data:
            data = self._conservation_cut(data,show_cut_info)
            data = self._leptons_cut(data,lepton_pT_cuts,lepton_eta_cuts,show_cut_info)
            data = self._Z_mass_cut(data,heavier_Z_cuts,lighter_Z_cuts,show_cut_info)

            reduced_data.append(data)

        return reduced_data
    
    def get_histogram(self,all_data,scalers,param,bins):
        total_hist = np.zeros(len(bins)-1)

        for data,scaler in zip(all_data,scalers):
            if param == 'm4l':
                hist, _ = np.histogram(self.get_m4l(data),bins=bins)
            else:
                hist, _ = np.histogram(data[:,self.field_indices[param]],bins=bins)
            total_hist += hist * scaler

        return total_hist

    def get_training_data(self,all_data,data_labels,fields,
                           balance_categories=True):
        if len(all_data) != len(data_labels):
            raise ValueError('Number of dataframes and label_sets must be the same')
        
        data = np.concatenate([
            data[:,[self.field_indices[field] for field in fields]] for data in all_data
        ],axis=0)
        labels = np.concatenate([
            np.array([label]*len(data)) for data,label in zip(all_data,data_labels)
        ],axis=0)

        if balance_categories:
            catergories = np.unique(labels)
            N_sample = min([np.sum(labels == category) for category in catergories])

            reduced_data = []
            reduced_labels = []

            for category in catergories:
                indices = np.where(labels == category)[0]
                reduced_indices = np.random.choice(indices,N_sample,replace=False)

                reduced_data.append(data[reduced_indices])
                reduced_labels.append(labels[reduced_indices])

            data = np.concatenate(reduced_data,axis=0)
            labels = np.concatenate(reduced_labels,axis=0)

        X = torch.tensor(data,dtype=torch.float32)
        Y = torch.tensor(labels,dtype=torch.float32)

        return X,Y

    def get_dataloaders(self,X,Y,
                         train_ratio=0.8,
                         batch_size=2048,num_workers=8,
                         seed=42):
        torch.manual_seed(seed)

        dataset = TensorDataset(X,Y)

        N_data = dataset.tensors[0].shape[0]
        N_train = int(train_ratio*N_data)
        N_val = N_data - N_train

        train_dataset, val_dataset = random_split(dataset,[N_train,N_val])

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,)

        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,)

        return train_loader, val_loader, N_data, N_train, N_val

if __name__ == '__main__':
    higgs_mc = [
        '../HiggsTo4L/MC/higgs2011.csv',
        '../HiggsTo4L/MC/higgs2012.csv',
    ]
    zz_mc = [
        '../HiggsTo4L/MC/zzto4mu2011.csv',
        '../HiggsTo4L/MC/zzto2mu2e2011.csv',
        '../HiggsTo4L/MC/zzto4e2011.csv',
        '../HiggsTo4L/MC/zzto4mu2012.csv',
        '../HiggsTo4L/MC/zzto2mu2e2012.csv',
        '../HiggsTo4L/MC/zzto4e2012.csv',
    ]

    data_files = higgs_mc + zz_mc

    four_leptons_data = FourLeptonsData()
    
    all_data = four_leptons_data.apply_basic_cuts([
        four_leptons_data._read_data(data_file) for data_file in data_files
    ])
    X,Y = four_leptons_data.get_training_data(
        all_data,
        data_labels = [1]*len(higgs_mc) + [0]*len(zz_mc),
        fields=[
            'E1','px1','py1','pz1','eta1','cos_phi1','sin_phi1',
            'E2','px2','py2','pz2','eta2','cos_phi2','sin_phi2',
            'E3','px3','py3','pz3','eta3','cos_phi3','sin_phi3',
            'E4','px4','py4','pz4','eta4','cos_phi4','sin_phi4',
        ]
    )
    train_loader, val_loader, N_data, N_train, N_val = four_leptons_data.get_dataloaders(X,Y)
    print(N_data,N_train,N_val)
    for x,y in train_loader:
        print(x.shape,y.shape)
        print(y.mean())
        break