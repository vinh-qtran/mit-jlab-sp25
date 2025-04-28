import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

#####################################################################################################

import os

import numpy as np

#####################################################################################################

class CosmicWatchMeasurement:
    field_idx = {
        3 : 'time',
        7 : 'temp',
        8 : 'pressure',
        9 : 'dead_time',
        10 : 'coincident'
    }

    def __init__(self,
                 data_file,
                 only_coincident=True,):
        
        self._read_data(data_file,only_coincident)
        self._get_rate()

    def _read_data(self,data_file,only_coincident):
        data = {
            field : [] for idx,field in self.field_idx.items()
        }

        with open(data_file, 'r') as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue
                
                line_data = line.split()

                for idx, field in self.field_idx.items():
                    data[field].append(float(line_data[idx]))

        for field in data:
            setattr(self,field,np.array(data[field]))

        self.total_time = self.time[-1]*1e-3

        self.dead_time = np.cumsum(self.dead_time)*1e-6
        self.time = self.time*1e-3 - self.dead_time

        self.live_time = self.time[-1]

        if only_coincident:
            mask = self.coincident == 1
            for field in data:
                setattr(self,field,getattr(self,field)[mask])

    def _read_

    def _get_rate(self):
        self.rate = (self.time.size-1) / np.sum(self.time[1:] - self.time[:-1])
        self.rate_err = self.rate / np.sqrt(self.time.size-1)

class AngleMeasurement:
    def __init__(self, data_dir):
        self._read_data(data_dir)

    def _read_data(self, data_dir):
        self.angles = np.array([])

        self.rates = np.array([])
        self.rates_err = np.array([])

        data_files = sorted(Path(data_dir).glob('*.txt'))
        for file in data_files:
            self.angles = np.append(self.angles, float(file.name.split('.')[0]))

            cosmic_watch_data = CosmicWatchMeasurement(file)

            self.rates = np.append(self.rates, cosmic_watch_data.rate)
            self.rates_err = np.append(self.rates_err, cosmic_watch_data.rate_err)

if __name__ == '__main__':
    # angle_measurement = AngleMeasurement('../data')

    # print(angle_measurement.angles)
    # print(angle_measurement.rates)
    # print(angle_measurement.rates/angle_measurement.rates_err)

    cosmic_watch_data = CosmicWatchMeasurement('../data/0.txt')
    print(cosmic_watch_data.total_time)
    print(cosmic_watch_data.live_time)