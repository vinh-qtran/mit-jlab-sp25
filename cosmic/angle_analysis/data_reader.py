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
        3 : 'times',
        7 : 'temperatures',
        8 : 'pressures',
        9 : 'dead_times',
        10 : 'coincident'
    }

    def __init__(self,
                 data_file,
                 only_coincident=True,):
        
        self._read_data(data_file)
        self.temperature_avg, self.temperature_std, self.temperature_max, self.temperature_min = self._get_average_param('temperatures')
        self.pressure_avg, self.pressure_std, self.pressure_max, self.pressure_min = self._get_average_param('pressures')

        if only_coincident:
            self._filter_data()

        self._get_rate()

    def _read_data(self,data_file):
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

        self.total_time = self.times[-1]*1e-3

        self.dead_times = np.cumsum(self.dead_times)*1e-6
        self.times = self.times*1e-3 - self.dead_times

        self.live_time = self.times[-1]

    def _filter_data(self):
        mask = self.coincident == 1
        for field in self.field_idx.values():
            setattr(self,field,getattr(self,field)[mask])

    def _get_rate(self):
        self.counts = self.times.size
        interval = self.times[-1] - self.times[0]

        self.rate = (self.counts-1) / interval
        self.rate_err = np.sqrt(self.counts-1) / interval

    def _get_average_param(self,param):
        param_avg = np.mean(getattr(self, param))
        param_std = np.std(getattr(self, param))

        param_max = np.max(getattr(self, param))
        param_min = np.min(getattr(self, param))

        return param_avg, param_std, param_max, param_min

class AngleMeasurement:
    data_fields = [
        'counts',
        'total_time',
        'live_time',

        'rate',
        'rate_err',

        'pressure_avg',
        'pressure_max',
        'pressure_min',
    ]

    def __init__(self, data_dir, atmosphere_file=None):
        self._read_data(data_dir)

        if atmosphere_file is not None:
            self._read_atmosphere(atmosphere_file)

    def _read_data(self, data_dir):
        self.angle = np.array([])
        for field in self.data_fields:
            setattr(self, field, np.array([]))

        data_files = sorted(Path(data_dir).glob('*.txt'))
        for file in data_files:
            self.angle = np.append(self.angle, float(file.name.split('.')[0]))

            cosmic_watch_data = CosmicWatchMeasurement(file)

            for field in self.data_fields:
                setattr(self, field, np.append(getattr(self, field), getattr(cosmic_watch_data, field)))

    def _read_atmosphere(self, atmosphere_file):
        field_idx = {
            0 : 'angle',
            2 : 'temperature_min',
            3 : 'temperature_avg',
            4 : 'temperature_max',
        }

        temperature_data = {
            field : [] for idx, field in field_idx.items()
        }

        with open(atmosphere_file, 'r') as f:
            for line in f.readlines():
                line_data = line.split()

                for idx, field in field_idx.items():
                    temperature_data[field].append(float(line_data[idx]) + 273.15)
        
        for field in temperature_data:
            if field != 'angle':
                setattr(self, field, np.array(temperature_data[field])[np.argsort(temperature_data['angle'])])

if __name__ == '__main__':
    angle_measurement = AngleMeasurement('../data','atmosphere.txt')

    for i in range(angle_measurement.angle.size):
        i = angle_measurement.angle.size - i - 1

        print(
            f'{90-int(angle_measurement.angle[i])} &',
            # f'${np.round(angle_measurement.total_time[i]/3600,2)} \pm {np.round(angle_measurement.total_time[i]/3600/24*1/60,2)}$ &',
            f'${np.round(angle_measurement.live_time[i]/3600,2)} \pm {np.round(angle_measurement.total_time[i]/3600/24*1/60,2)}$ &',
            f'${int(angle_measurement.counts[i])} \pm {int(angle_measurement.total_time[i]/3600/24*5)}$ &',
            f'${np.round(angle_measurement.temperature_avg[i],1)}^'+'{'+f'+{np.round(angle_measurement.temperature_max[i]-angle_measurement.temperature_avg[i],1)}'+'}_{'+f'-{np.round(angle_measurement.temperature_avg[i]-angle_measurement.temperature_min[i],1)}'+'}$ &',
            f'${np.round(angle_measurement.pressure_avg[i]/100,1)}^'+'{'+f'+{np.round(angle_measurement.pressure_max[i]/100-angle_measurement.pressure_avg[i]/100,1)}'+'}_{'+f'-{np.round(angle_measurement.pressure_avg[i]/100-angle_measurement.pressure_min[i]/100,1)}'+'}$ \\\\'
        )

        print()