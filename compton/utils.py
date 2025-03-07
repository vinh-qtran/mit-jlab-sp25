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


####################################################################################################################

'''
Base MCA Data Class
'''

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
            clear_peaks_idx.pop(0)
        return clear_peaks_idx, clear_valleys_idx

    def _get_fitting_boundaries(self,bins,kdes,clear_valleys_idx,
                                lower_feature_bin,upper_feature_bin,
                                threshold=None,threshold_ratio=1):
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
            threshold = max(kdes[lower_valley_idx],kdes[upper_valley_idx]) * threshold_ratio

        def _get_boundary(valley_idx,upper=False):
            '''
            Helper function to get the boundary of the fitting region.
            '''

            if kdes[valley_idx] >= threshold:
                return valley_idx
            
            for i in range(upper_valley_idx - lower_valley_idx):
                i = upper_valley_idx - i if upper else lower_valley_idx + i

                if kdes[i] > threshold:
                    return i

            return None

        lower_idx = _get_boundary(lower_valley_idx,upper=False)
        upper_idx = _get_boundary(upper_valley_idx,upper=True)

        return lower_idx, upper_idx
    
    def _fit_peaks(self,bins,counts,lower_idx,upper_idx,
                   fitter_class,initial_guess,n_gaussian=1):
        '''
        Fit features and return the mean(s) and error(s) of the gaussian-approximated peak(s).

        Input:
            bins: 1D array, bin indices
            counts: 1D array, counts
            fitting_idx: 1D array, indices of the fitting region
            fitter_class: class, fitter class
            initial_guess: 1D array, initial guess of the gaussian parameters

        Output:
            peak_mu: float, mean of the gaussian-approximated peak(s)
            peak_mu_err: float, error of the mean of the gaussian-approximated peak(s)
            fitting_bins: 1D array, bin indices of the fitting region
            gaussian_counts: 1D array, counts of the gaussian-approximated peak(s)
        '''
        fitting_idx = np.arange(lower_idx,upper_idx+1)

        fitting_bins = bins[fitting_idx]

        count_normalizing_factor = np.sum(counts) * (bins[1] - bins[0])
        fitting_counts = counts[fitting_idx] / count_normalizing_factor
        fitting_counts_err = np.sqrt(counts)[fitting_idx] / count_normalizing_factor

        fitter = fitter_class(fitting_bins,fitting_counts,fitting_counts_err,initial_guess)
        fitting_result = fitter.fit()

        peak_mu = fitting_result['params'][:n_gaussian]
        peak_mu_err = fitting_result['e_params'][:n_gaussian] if fitting_result['e_params'] is not None else np.array([np.nan]*n_gaussian)
        fitted_counts = fitter._get_model(fitting_bins,fitting_result['params']) * count_normalizing_factor

        return peak_mu, peak_mu_err, fitting_bins, fitted_counts
    
    def _fit_single_peak(self,
                         bins,counts,kdes,
                         clear_peaks_idx,clear_valleys_idx,
                         approx_line_bin=None,
                         threshold_ratio=1/3):
        '''
        Fit a single peak and return the mean and error of the gaussian-approximated peak if the approximated peak bin is provided.

        Input:
            bins: 1D array, bin indices
            counts: 1D array, counts
            kdes: 1D array, smoothed density estimations
            clear_peaks_idx: 1D array, indices of the clear peaks
            clear_valleys_idx: 1D array, indices of the clear valleys
            approx_line_bin: int, approximate bin of the peak

        Output:
            peak_mu: float, mean of the gaussian-approximated peak
            peak_mu_err: float, error of the mean of the gaussian-approximated peak
            fitting_bins: 1D array, bin indices of the fitting region
            gaussian_counts: 1D array, counts of the gaussian-approximated peak
        '''
        if approx_line_bin is None:
            return np.array([np.nan]), np.array([np.nan]), np.array([]), np.array([])

        # Find the fitting region
        approx_peak_idx = clear_peaks_idx[
            np.argmin(np.abs(bins[clear_peaks_idx] - approx_line_bin))
        ]
        approx_peak_bin = bins[approx_peak_idx]
        approx_peak_kde = kdes[approx_peak_idx]

        lower_idx, upper_idx = self._get_fitting_boundaries(bins,kdes,clear_valleys_idx,
                                                            approx_peak_bin,approx_peak_bin,
                                                            threshold=approx_peak_kde*threshold_ratio)
        
        # Get Gaussian initial guess
        mu_guess = approx_peak_bin
        sigma_guess = (bins[upper_idx] - bins[lower_idx]) / 4
        A_guess = approx_peak_kde * sigma_guess * np.sqrt(2*np.pi)
        c_guess = 0

        initial_guess = [mu_guess,sigma_guess,A_guess,c_guess]

        # Fit
        return self._fit_peaks(bins,counts,lower_idx,upper_idx,GaussianFitter,initial_guess)


####################################################################################################################

'''
Fitter Classes
'''
class LinearFitter(fitting.BaseFitter):
    pass

class GaussianFitter(fitting.BaseFitter):
    def __init__(self,x,y,yerr,initial_guess,no_baseline=False):
        self.x = x
        self.y = y
        self.yerr = yerr

        super().__init__(x,y,yerr)

        self.initial_guess = initial_guess

        self.no_baseline = no_baseline

    def _get_initial_guess(self):
        return self.initial_guess

    def _get_model(self,x,params):
        mu, sigma, A, c = params
        if self.no_baseline:
            c = 0
        return A / np.sqrt(2*np.pi) / sigma * np.exp(-1/2 * ((x - mu) / sigma)**2) + c

class DoubleGaussianFitter(GaussianFitter):
    def __init__(self,x,y,yerr,initial_guess):
        super().__init__(x,y,yerr,initial_guess)

    def _get_model(self,x,params):
        mu1, mu2, sigma1, sigma2, A1, A2, c = params
        return A1 / np.sqrt(2*np.pi) / sigma1 * np.exp(-1/2 * ((x - mu1) / sigma1)**2) + \
               A2 / np.sqrt(2*np.pi) / sigma2 * np.exp(-1/2 * ((x - mu2) / sigma2)**2) + c
    
class TripleGaussianFitter(GaussianFitter):
    def __init__(self,x,y,yerr,initial_guess):
        super().__init__(x,y,yerr,initial_guess)

    def _get_model(self,x,params):
        mu1, mu2, mu3, sigma1, sigma2, sigma3, A1, A2, A3, c = params
        return A1 / np.sqrt(2*np.pi) / sigma1 * np.exp(-1/2 * ((x - mu1) / sigma1)**2) + \
               A2 / np.sqrt(2*np.pi) / sigma2 * np.exp(-1/2 * ((x - mu2) / sigma2)**2) + \
               A3 / np.sqrt(2*np.pi) / sigma3 * np.exp(-1/2 * ((x - mu3) / sigma3)**2) + c
    
gaussian_fitter_classes = {
    1 : GaussianFitter,
    2 : DoubleGaussianFitter,
    3 : TripleGaussianFitter,
}


####################################################################################################################

'''
MCA Calibration Class
'''                 
class MCACalibration(MCAData):
    def __init__(self,
                 na_22_data_file,ba_133_data_file,kernel_bw=5,
                 na_22_energy=511,na_22_approx_line_bin=1400,
                 ba_133_energies=[[53.15,79.60,81.00],[160.61],[276.40,302.85,356.01]],
                 ba_133_peak_ratios=[[0.3,0.1,1],[1],[0.2,0.4,1]],
                 cs_137_energy=661.7,cs_137_approx_line_bin=None,
                 non_calib_energies=[276.40]):
        self.na_22_energy = na_22_energy
        self.ba_133_energies = ba_133_energies
        self.cs_137_energy = cs_137_energy

        self.na_22_peak_mu, self.na_22_peak_mu_err, self.na_22_fitting_bins, self.na_22_fitted_counts = self._fit_na_22_peak(na_22_data_file,kernel_bw,na_22_approx_line_bin)

        self.ba_133_peaks_mu, self.ba_133_peaks_mu_err, self.ba_133_fitting_bins, self.ba_133_fitted_counts, \
        self.cs_137_peak_mu, self.cs_137_peak_mu_err, self.cs_137_fitting_bins, self.cs_137_fitted_counts = self._fit_ba_133_peaks(ba_133_data_file,kernel_bw,ba_133_energies,ba_133_peak_ratios,cs_137_approx_line_bin)

        non_calib_energies = non_calib_energies if cs_137_approx_line_bin is not None else non_calib_energies + [cs_137_energy]
        self._get_calib_stats(non_calib_energies)

    def _fit_na_22_peak(self,
                        na_22_data_file,kernel_bw,
                        na_22_approx_line_bin):
        '''
        Fit the 511eV feature of Na-22 and return the mean and error of the gaussian-approximated peak.

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
        bins, counts, _, _ = self._read_data(na_22_data_file)
        bins, kdes, kdes_err = self._kde_smooth_data(bins,counts,kernel_bw)

        # Find peaks and valleys
        clear_peaks_idx, clear_valleys_idx = self._find_peaks_and_valleys(bins,kdes,kdes_err)

        # Fit
        return self._fit_single_peak(bins,counts,kdes,clear_peaks_idx,clear_valleys_idx,na_22_approx_line_bin)

    def _fit_ba_133_peaks(self,
                          ba_133_data_file,kernel_bw,
                          ba_133_energies,ba_133_peak_ratios,
                          cs_137_approx_line_bin):

        # Read data
        bins, counts, _, _ = self._read_data(ba_133_data_file)
        bins, kdes, kdes_err = self._kde_smooth_data(bins,counts,kernel_bw)

        # Find peaks and valleys
        clear_peaks_idx, clear_valleys_idx = self._find_peaks_and_valleys(bins,kdes,kdes_err)

        # Fit peaks of Ba-133
        ba_133_peaks_mu = []
        ba_133_peaks_mu_err = []
        ba_133_fitting_bins = []
        ba_133_fitted_counts = []

        for bundled_energies,peak_ratios in zip(ba_133_energies,ba_133_peak_ratios):
            # Find the fitting region
            approx_peaks_idx = self.na_22_peak_mu / self.na_22_energy * np.array(bundled_energies)

            lower_idx, upper_idx = self._get_fitting_boundaries(bins,kdes,clear_valleys_idx,
                                                                approx_peaks_idx[0],approx_peaks_idx[-1])

            n_gaussian = len(bundled_energies)
            # Get Gaussian initial guess
            mu_guess = approx_peaks_idx
            sigma_guess = (bins[upper_idx] - bins[lower_idx]) / 8
            A_guess = kdes[np.argmin(np.abs(bins - approx_peaks_idx[np.argmax(peak_ratios)]))] * sigma_guess * np.sqrt(2*np.pi) * np.array(peak_ratios)
            c_guess = 0

            initial_guess = [*mu_guess] + [sigma_guess]*n_gaussian + [*A_guess,c_guess]

            # Fit
            peaks_mu, peaks_mu_err, fitting_bins, fitted_counts = self._fit_peaks(bins,counts,lower_idx,upper_idx,gaussian_fitter_classes[n_gaussian],initial_guess,n_gaussian=n_gaussian)

            ba_133_peaks_mu.append(peaks_mu)
            ba_133_peaks_mu_err.append(peaks_mu_err)
            ba_133_fitting_bins.append(fitting_bins)
            ba_133_fitted_counts.append(fitted_counts)

        # Fit peaks of Cs-137 (if the approximate line is provided)
        cs_137_peak_mu, cs_137_peak_mu_err, cs_137_fitting_bins, cs_137_fitted_counts = self._fit_single_peak(bins,counts,kdes,clear_peaks_idx,clear_valleys_idx,cs_137_approx_line_bin)

        return ba_133_peaks_mu, ba_133_peaks_mu_err, ba_133_fitting_bins, ba_133_fitted_counts, \
               cs_137_peak_mu, cs_137_peak_mu_err, cs_137_fitting_bins, cs_137_fitted_counts
    
    def _get_calib_stats(self,non_calib_energies):
        '''
        Get the calibration statistics.
        '''

        self.calib_bins = np.concatenate([self.na_22_peak_mu] + self.ba_133_peaks_mu + [self.cs_137_peak_mu])
        self.calib_bins_err = np.concatenate([self.na_22_peak_mu_err] + self.ba_133_peaks_mu_err + [self.cs_137_peak_mu_err])
        self.calib_energies = np.concatenate([np.array([self.na_22_energy])] + self.ba_133_energies + [np.array([self.cs_137_energy])])

        sorting_order = np.argsort(self.calib_energies)
        mask = np.ones(self.calib_bins.size,dtype=bool)
        for energy in non_calib_energies:
            mask[self.calib_energies[sorting_order] == energy] = False
        self.calib_bins = self.calib_bins[sorting_order][mask]
        self.calib_bins_err = self.calib_bins_err[sorting_order][mask]
        self.calib_energies = self.calib_energies[sorting_order][mask]

        scaler_bayes_gaussian = fitting.BayesianGaussian(self.calib_energies/self.calib_bins)
        self.energy_scaler, self.energy_scaler_err = scaler_bayes_gaussian.mu, scaler_bayes_gaussian.sigma
        self.energy_scaler_mse = np.mean((self.calib_energies - self.calib_bins*self.energy_scaler)**2/self.calib_energies**2)

class MCACompton(MCACalibration):
    def __init__(self,data_file):
        pass