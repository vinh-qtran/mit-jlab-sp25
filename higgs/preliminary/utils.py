import os
import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

import numpy as np
import torch

import pandas as pd

from modules.nn import pca

import importlib
importlib.reload(pca)

##########################################################################################

class FourLeptonsReader:
    def __init__(self,
                 data_files,scalers):
        self.data_files = data_files
        self.scalers = scalers

        self.dfs = [self._read_data(file) for file in data_files]

    def _read_data(self,data_file):
        df = pd.read_csv(data_file)

        for i in range(4):
            df[f'pT{i+1}'] = np.sqrt(df[f'px{i+1}']**2 + df[f'py{i+1}']**2)

        df['m4l'] = np.sqrt(
            (df['E1']+df['E2']+df['E3']+df['E4'])**2 - \
            (df['px1']+df['px2']+df['px3']+df['px4'])**2 - \
            (df['py1']+df['py2']+df['py3']+df['py4'])**2 - \
            (df['pz1']+df['pz2']+df['pz3']+df['pz4'])**2
        )

        return df
    
    def _conservation_cut(self,df,show_cut_info=True):
        mask = np.sum(df[['PID1','PID2','PID3','PID4']],axis=1) == 0

        if show_cut_info:
            print(f' Conservation cut: {mask.sum()} events passed out of {len(mask)} ({100*mask.sum()/len(mask):.0f}%)')

        return df[mask]
    
    def _leptons_cut(self,df,lepton_pT_cuts=[7,5],lepton_eta_cuts=[2.5,2.4],show_cut_info=True):
        def _single_lepton_cut(PID,pT,eta):
            electron_mask = np.logical_and(
                np.abs(PID) == 11,
                np.logical_and(pT > lepton_pT_cuts[0], np.abs(eta) < lepton_eta_cuts[0])
            )

            muon_mask = np.logical_and(
                np.abs(PID) == 13,
                np.logical_and(pT > lepton_pT_cuts[1], np.abs(eta) < lepton_eta_cuts[1])
            )

            return np.logical_or(electron_mask,muon_mask).to_numpy().reshape((-1,1))
        
        mask = np.concatenate([
            _single_lepton_cut(df[f'PID{i+1}'],df[f'pT{i+1}'],df[f'eta{i+1}']) for i in range(4)
        ],axis=1).all(axis=1)

        if show_cut_info:
            print(f' Leptons cut: {mask.sum()} events passed out of {len(mask)} ({100*mask.sum()/len(mask):.0f}%)')

        return df[mask]
    
    def _Z_mass_cut(self,df,heavier_Z_cuts=[40,120],lighter_Z_cuts=[12,120],show_cut_info=True):
        def _get_paired_Z_mass(paired_ids):
            return np.sqrt(
                (df[f'E{paired_ids[0]}'] + df[f'E{paired_ids[1]}'])**2 - \
                (df[f'px{paired_ids[0]}'] + df[f'px{paired_ids[1]}'])**2 - \
                (df[f'py{paired_ids[0]}'] + df[f'py{paired_ids[1]}'])**2 - \
                (df[f'pz{paired_ids[0]}'] + df[f'pz{paired_ids[1]}'])**2
            )

        masks = []
        for i in range(2,5):
            mZ1 = _get_paired_Z_mass([1,i])
            mZ2 = _get_paired_Z_mass([j for j in range(2,5) if j != i])

            good_pair_mask = df['PID1'] + df[f'PID{i}'] == 0

            heavier_Z1_mask = np.logical_and(
                np.logical_and(mZ1 > heavier_Z_cuts[0], mZ1 < heavier_Z_cuts[1]),
                np.logical_and(mZ2 > lighter_Z_cuts[0], mZ2 < lighter_Z_cuts[1])
            )

            lighter_Z1_mask = np.logical_and(
                np.logical_and(mZ1 > lighter_Z_cuts[0], mZ1 < lighter_Z_cuts[1]),
                np.logical_and(mZ2 > heavier_Z_cuts[0], mZ2 < heavier_Z_cuts[1])
            )

            masks.append(np.logical_and(good_pair_mask,np.logical_or(heavier_Z1_mask,lighter_Z1_mask)).to_numpy().reshape((-1,1)))

        mask = np.concatenate(masks,axis=1).any(axis=1)

        if show_cut_info:
            print(f' Z cut: {mask.sum()} events passed out of {len(mask)} ({100*mask.sum()/len(mask):.0f}%)')

        return df[mask]
    
    def apply_basic_cuts(self,dfs,
                         lepton_pT_cuts=[7,5],lepton_eta_cuts=[2.5,2.4],
                         heavier_Z_cuts=[40,120],lighter_Z_cuts=[12,120],
                         show_cut_info=True):
        reduced_dfs = []

        for df in dfs:
            df = self._conservation_cut(df,show_cut_info)
            df = self._leptons_cut(df,lepton_pT_cuts,lepton_eta_cuts,show_cut_info)
            df = self._Z_mass_cut(df,heavier_Z_cuts,lighter_Z_cuts,show_cut_info)

            reduced_dfs.append(df)

        return reduced_dfs
    
    def get_histogram(self,dfs,params,bins):
        total_hist = np.zeros(len(bins)-1)

        for df,scaler in zip(dfs,self.scalers):
            hist, _ = np.histogram(df[params],bins=bins)
            total_hist += hist * scaler

        return total_hist
    
if __name__ == '__main__':
    pass