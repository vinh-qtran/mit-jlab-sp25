import os
import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

from modules import fitting

import numpy as np

# DATA READER

class MCAData:
    time_idx = 9
    data_idx = 12

    def _read_data(self,data_file):
        '''
        Read data from a .Spe file and return the bin indices, the corresponding counts, the counting time, and the total time.

        Input:
            data_file: string, path to the .Spe file

        Output:
            bins: 1D array, bin indices
            counts: 1D array, counts
            count_time: float, count time
            total_time: float, total time
        '''

        bins = np.array([])
        counts = np.array([])

        with open(data_file,'r') as f:
            for i,line in enumerate(f.readlines()):
                if i == self.time_idx:
                    count_time, total_time = np.array(line.split(),dtype=float)
                elif i >= self.data_idx:
                    if line.startswith('$'):
                        break
                    bins = np.append(bins,i-self.data_idx)
                    counts = np.append(counts,float(line))
        
        if counts.size != 2048:
            raise ValueError('Incorrect number of bins')

        return bins, counts, count_time, total_time
    
    def _kde_smooth_data(self,bins,counts,bw=1):
        '''
        Smooth data using a kernel density estimate with the bandwidth of bw.

        Input:
            bins: 1D array, bin indices
            counts: 1D array, counts
            bw: float, bandwidth of the kernel

        Output:
            bins: 1D array, bin indices
            kdes: 1D array, smoothed density estimations
            kdes_err: 1D array, error of the smoothed density estimations
        '''

        bins_matrix = np.vstack([bins] * bins.size)

        density_matrix = 1/(np.sqrt(2*np.pi)*bw) * np.exp(-1/2 * ((bins_matrix - bins_matrix.T) / bw)**2)
        
        kdes = np.dot(density_matrix.T,counts)
        kdes_err = np.dot(density_matrix.T,np.sqrt(counts))

        normalizing_factor = np.sum(kdes) * (bins[1] - bins[0])

        return bins, kdes/normalizing_factor, kdes_err/normalizing_factor

    def _find_peaks_and_valleys(self,bins,kdes,kdes_err):
        '''
        Find the peaks and valleys in the smoothed data on the prinple of first derivative, with statistical error considered.

        Input:
            bins: 1D array, bin indices
            kdes: 1D array, smoothed density estimations
            kdes_err: 1D array, error of the smoothed density estimations

        Output:
            clear_peaks_idx: 1D array, indices of the clear peaks
            clear_valleys_idx: 1D array, indices of the clear valleys
        '''

        # First derivative principle
        d_kdes = np.diff(kdes)
        d2_kdes = np.diff(d_kdes)

        peak_or_valley = d_kdes[:-1] * d_kdes[1:] < 0

        accelerating_slop = d2_kdes > 0
        decelerating_slop = d2_kdes < 0

        peaks_idx = np.where(np.logical_and(peak_or_valley,decelerating_slop))[0] + 1
        valleys_idx = np.where(np.logical_and(peak_or_valley,accelerating_slop))[0] + 1

        # Statistical error consideration
        peak_or_valley_idx = np.sort(np.append(peaks_idx,valleys_idx))
        peak_and_valley_kdes = kdes[peak_or_valley_idx]
        peak_and_valley_kdes_err = kdes_err[peak_or_valley_idx]

        clear_idx = [[],[]]

        bundled_idx = []
        last_clear_is_peak = None

        for i in range(peak_or_valley_idx.size-1):
            right_deviation = np.abs(peak_and_valley_kdes[i+1] - peak_and_valley_kdes[i]) / np.sqrt(peak_and_valley_kdes_err[i+1]**2 + peak_and_valley_kdes_err[i]**2)

            bundled_idx.append(peak_or_valley_idx[i])

            if right_deviation > 1:
                bundled_is_peak = peak_or_valley_idx[i] in peaks_idx
                if i and bundled_is_peak == last_clear_is_peak:
                    continue

                clear_idx[int(not bundled_is_peak)].append(
                    bundled_idx[np.argmax(kdes[bundled_idx])] if bundled_is_peak else bundled_idx[np.argmin(kdes[bundled_idx])]
                )

                bundled_idx = []
                last_clear_is_peak = bundled_is_peak

        clear_peaks_idx, clear_valleys_idx = clear_idx
        if last_clear_is_peak:
            clear_valleys_idx.append(bins.size-1)
        if min(clear_peaks_idx) < min(clear_valleys_idx):
            clear_valleys_idx.insert(0,0)
        return clear_peaks_idx, clear_valleys_idx

    def _get_fitting_boundaries(self,bins,kdes,clear_valleys_idx,
                                lower_feature_bin,upper_feature_bin,
                                threshold=None):
        '''
        Get the fitting boundaries of a (or multiple) peak(s).

        Input:
            bins: 1D array, bin indices
            kdes: 1D array, smoothed density estimations
            clear_valleys_idx: 1D array, indices of the clear valleys
            lower_feature_bin: float, lower bound of the feature
            upper_feature_bin: float, upper bound of the feature
            threshold: float, threshold of the peak

        Output:
            lower_idx: int, lower boundary of the fitting region
            upper_idx: int, upper boundary of the fitting region
        '''

        # Get the closest valleys to the feature
        clear_valleys_bins = bins[clear_valleys_idx]
        clear_valleys_arange = np.arange(len(clear_valleys_idx))
        
        lower_valley_idx = clear_valleys_idx[np.max(clear_valleys_arange[clear_valleys_bins < lower_feature_bin])]
        upper_valley_idx = clear_valleys_idx[np.min(clear_valleys_arange[clear_valleys_bins > upper_feature_bin])]

        # Get the fitting region boundaries, satisfying the threshold
        if threshold is None:
            threshold = max(kdes[lower_valley_idx],kdes[upper_valley_idx])

        def _get_boundary(valley_idx,upper=False):
            '''
            Helper function to get the boundary of the fitting region.
            '''

            if kdes[valley_idx] > threshold:
                return valley_idx
            
            for i in range(upper_valley_idx - lower_valley_idx):
                i = upper_valley_idx - i if upper else lower_valley_idx + i

                if kdes[i] > threshold:
                    return i

            return None

        lower_idx = _get_boundary(lower_valley_idx,upper=False)
        upper_idx = _get_boundary(upper_valley_idx,upper=True)

        return lower_idx, upper_idx


class GaussianFitter(fitting.BaseFitter):
    def __init__(self,x,y,yerr,initial_guess):
        self.x = x
        self.y = y
        self.yerr = yerr

        super().__init__(x,y,yerr)

        self.initial_guess = initial_guess

    def _get_initial_guess(self):
        return self.initial_guess

    def _get_model(self,x,params):
        mu, sigma, A, c = params
        return A / np.sqrt(2*np.pi) / sigma * np.exp(-1/2 * ((x - mu) / sigma)**2) + c

class TripleGaussianFitter(GaussianFitter):
    def __init__(self,x,y,yerr,initial_guess):
        super().__init__(x,y,yerr,initial_guess)

    def _get_model(self,x,params):
        mu1, sigma1, mu2, sigma2, mu3, sigma3, A1, A2, A3, c = params
        return A1 / np.sqrt(2*np.pi) / sigma1 * np.exp(-1/2 * ((x - mu1) / sigma1)**2) + \
               A2 / np.sqrt(2*np.pi) / sigma2 * np.exp(-1/2 * ((x - mu2) / sigma2)**2) + \
               A3 / np.sqrt(2*np.pi) / sigma3 * np.exp(-1/2 * ((x - mu3) / sigma3)**2) + c
    
                    
class MCACalibration(MCAData):
    def __init__(self,
                 na_22_data_file,ba_133_data_file,
                 na_22_energy=511,na_22_approx_line_bin=1400,
                 ba_133_energies=[[53.15,79.60,81.00],[276.40,302.85,356.01]],
                 ba_133_peak_ratios=[[0.2,0.3,1],[0.3,0.4,1]],
                 kernel_bw=5,):
        self.na_22_energy = na_22_energy

        self.na_22_peak_mu, self.na_22_peak_mu_err, _, _ = self._fit_na_22_peak(na_22_data_file,na_22_approx_line_bin,kernel_bw)

    def _fit_na_22_peak(self,na_22_data_file,na_22_approx_line_bin,kernel_bw):
        '''
        Fit the 511eV feature of Na-22 and return the mean and the error of the gaussian-approximated peak.

        Input:
            na_22_data_file: string, path to the .Spe file of Na-22
            na_22_approx_line_bin: int, approximate bin of the 511eV feature
            kernel_bw: float, bandwidth of the kernel

        Output:
            peak_mu: float, mean of the gaussian-approximated peak
            peak_mu_err: float, error of the mean of the gaussian-approximated peak
            fitting_bins: 1D array, bin indices of the fitting region
            gaussian_counts: 1D array, counts of the gaussian-approximated peak
        '''

        # Read data
        bins, counts, count_time, total_time = self._read_data(na_22_data_file)
        bins, kdes, kdes_err = self._kde_smooth_data(bins,counts,kernel_bw)

        # Find peaks and valleys
        clear_peaks_idx, clear_valleys_idx = self._find_peaks_and_valleys(bins,kdes,kdes_err)

        # Find the fitting region
        approx_peak_idx = clear_peaks_idx[
            np.argmin(np.abs(bins[clear_peaks_idx] - na_22_approx_line_bin))
        ]
        approx_peak_bin = bins[approx_peak_idx]
        approx_peak_kde = kdes[approx_peak_idx]

        lower_idx, upper_idx = self._get_fitting_boundaries(bins,kdes,clear_valleys_idx,
                                                            approx_peak_bin,approx_peak_bin,
                                                            threshold=approx_peak_kde/3)
        fitting_idx = np.arange(lower_idx,upper_idx+1)
        
        # Get Gaussian initial guess
        mu_guess = approx_peak_bin
        sigma_guess = (bins[upper_idx] - bins[lower_idx]) / 2
        A_guess = approx_peak_kde * sigma_guess * np.sqrt(2*np.pi)
        c_guess = 0

        initial_guess = [mu_guess,sigma_guess,A_guess,c_guess]

        # Fit
        fitting_bins = bins[fitting_idx]

        count_normalizing_factor = np.sum(counts) * (bins[1] - bins[0])
        fitting_counts = counts[fitting_idx] / count_normalizing_factor
        fitting_counts_err = np.sqrt(counts)[fitting_idx] / count_normalizing_factor

        fitter = GaussianFitter(fitting_bins,fitting_counts,fitting_counts_err,initial_guess)
        fitting_result = fitter.fit()

        peak_mu = fitting_result['params'][0]
        peak_mu_err = fitting_result['e_params'][0]
        gaussian_counts = fitter._get_model(fitting_bins,fitting_result['params']) * count_normalizing_factor

        return peak_mu, peak_mu_err, fitting_bins, gaussian_counts

    def _fit_ba_133_peaks(self,ba_133_data_file,ba_133_energies,ba_133_peak_ratios,kernel_bw):

        # Read data
        bins, counts, count_time, total_time = self._read_data(ba_133_data_file)
        bins, kdes, kdes_err = self._kde_smooth_data(bins,counts,kernel_bw)

        # Find peaks and valleys
        clear_peaks_idx, clear_valleys_idx = self._find_peaks_and_valleys(bins,kdes,kdes_err)