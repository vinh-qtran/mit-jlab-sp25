import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

######################################################################################################

import numpy as np

from scipy.integrate import trapezoid

import json

#######################################################################################################

class CrossSection:
    def __init__(self,cross_section_file):
        self._read_cross_section(cross_section_file)

        self.dsigma_dtheta = self._get_cross_section_theta_dependency()

    def _read_cross_section(self,cross_section_file):
        with open(cross_section_file, 'r') as f:
            data = json.load(f)

        self.theta_bins = np.array(data['thetas'])
        self.phi_bins = np.array(data['phis'])
        self.dsigma_dOmega_grid = np.array(data['dsigma_dOmega'])

    def _get_cross_section_theta_dependency(self):
        return trapezoid(
            self.dsigma_dOmega_grid,
            dx = self.phi_bins[1] - self.phi_bins[0],
            axis = 1
        ) * np.sin(self.theta_bins)