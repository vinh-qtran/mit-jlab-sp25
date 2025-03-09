import os
import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

from modules import fitting
import importlib
importlib.reload(fitting)

import numpy as np


####################################################################################################################

'''
Fitter Classes
'''
class NoUncertaintyLinearFitter(fitting.BaseFitter):
    '''
    Fitter class for linear fitting without uncertainties.
    '''

    def __init__(self,x,y,initial_guess):
        super().__init__(x,y,np.ones_like(y))

        self.initial_guess = initial_guess

    def _get_initial_guess(self):
        return self.initial_guess
    
    def _get_model(self,x,params):
        a, b = params
        return a*x + b
    
    def fit(self):
        a, b = super().fit()['params']

        a_s = (self.y - b) / self.x
        b_s = self.y - a * self.x

        a_err = np.std(a_s)
        b_err = np.std(b_s)

        return a, b, a_err, b_err

class GaussianFitter(fitting.BaseFitter):
    '''
    Gaussian fitter class.
    '''

    def __init__(self,x,y,yerr,initial_guess):
        super().__init__(x,y,yerr)

        self.initial_guess = initial_guess

    def _get_initial_guess(self):
        return self.initial_guess

    def _get_model(self,x,params):
        mu, sigma, A, *poly_params = params
        return A / np.sqrt(2*np.pi) / sigma * np.exp(-1/2 * ((x - np.abs(mu)) / sigma)**2) + np.polyval(poly_params,x)
    
class GaussianPoissFitter(fitting.BasePoissonFitter):
    '''
    Gaussian fitter class with Poisson statistics.
    '''

    def __init__(self,x,y,yerr,initial_guess):
        super().__init__(x,y)

        self.initial_guess = initial_guess

    def _get_initial_guess(self):
        return self.initial_guess
    
    def _get_model(self,x,params):
        mu, sigma, A, *poly_params = params
        return A / np.sqrt(2*np.pi) / sigma * np.exp(-1/2 * ((x - np.abs(mu)) / sigma)**2) + np.polyval(poly_params,x)

class DoubleGaussianFitter(GaussianFitter):
    '''
    Gaussian fitter class for double peaks.
    '''

    def __init__(self,x,y,yerr,initial_guess):
        super().__init__(x,y,yerr,initial_guess)

    def _get_model(self,x,params):
        mu1, mu2, sigma1, sigma2, A1, A2, *poly_params = params
        return A1 / np.sqrt(2*np.pi) / sigma1 * np.exp(-1/2 * ((x - np.abs(mu1)) / sigma1)**2) + \
               A2 / np.sqrt(2*np.pi) / sigma2 * np.exp(-1/2 * ((x - np.abs(mu2)) / sigma2)**2) + \
               np.polyval(poly_params,x)
    
class TripleGaussianFitter(GaussianFitter):
    '''
    Gaussian fitter class for triple peaks.
    '''

    def __init__(self,x,y,yerr,initial_guess):
        super().__init__(x,y,yerr,initial_guess)

    def _get_model(self,x,params):
        mu1, mu2, mu3, sigma1, sigma2, sigma3, A1, A2, A3, *poly_params = params
        return A1 / np.sqrt(2*np.pi) / sigma1 * np.exp(-1/2 * ((x - np.abs(mu1)) / sigma1)**2) + \
               A2 / np.sqrt(2*np.pi) / sigma2 * np.exp(-1/2 * ((x - np.abs(mu2)) / sigma2)**2) + \
               A3 / np.sqrt(2*np.pi) / sigma3 * np.exp(-1/2 * ((x - np.abs(mu3)) / sigma3)**2) + \
               np.polyval(poly_params,x)
    
gaussian_fitter_classes = {
    1 : GaussianFitter,
    2 : DoubleGaussianFitter,
    3 : TripleGaussianFitter,
}


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
            bins: 1D array, bin numbers
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

    def _get_fitting_boundaries(self,bins,kdes,clear_peaks_idx,clear_valleys_idx,
                                lower_feature_bin,upper_feature_bin,
                                threshold=None,threshold_ratio=1,
                                outward=False):
        '''
        Get the fitting boundaries of a (or multiple) peak(s).

        Input:
            bins: 1D array, bin numbers
            kdes: 1D array, smoothed density estimations
            clear_valleys_idx: 1D array, indices of the clear valleys
            lower_feature_bin: float, lower bound of the feature
            upper_feature_bin: float, upper bound of the feature
            threshold: float, threshold of the peak
            threshold_ratio: float, ratio of the threshold
            outward: bool, whether to fit outward

        Output:
            lower_idx: int, lower boundary of the fitting region
            upper_idx: int, upper boundary of the fitting region
        '''

        # Get the closest peaks and valleys to the feature
        def _find_closest_extremum(feature_bin,clear_extrema_idx,left=True):
            clear_extrema_bins = bins[clear_extrema_idx]
            clear_extrema_arange = np.arange(len(clear_extrema_idx))
                                             
            if left:
                return clear_extrema_idx[np.max(clear_extrema_arange[clear_extrema_bins <= feature_bin])]
            return clear_extrema_idx[np.min(clear_extrema_arange[clear_extrema_bins >= feature_bin])]
            
        lower_peak_idx = _find_closest_extremum(lower_feature_bin,clear_peaks_idx,left=False)
        upper_peak_idx = _find_closest_extremum(upper_feature_bin,clear_peaks_idx,left=True)
        
        lower_valley_idx = _find_closest_extremum(lower_feature_bin,clear_valleys_idx,left=True)
        upper_valley_idx = _find_closest_extremum(upper_feature_bin,clear_valleys_idx,left=False)

        # Get the fitting region boundaries, satisfying the threshold
        if not threshold:
            threshold = max(kdes[lower_valley_idx],kdes[upper_valley_idx]) * threshold_ratio

        def _get_boundary(start_idx,end_idx):
            '''
            Helper function to get the boundary of the fitting region.
            '''
            sign = -1 if outward else 1
            
            for i in range(start_idx,end_idx,1 if start_idx < end_idx else -1):
                if sign*kdes[i] > sign*threshold:
                    return i

            return None
        
        lower_idx = _get_boundary(lower_peak_idx,lower_valley_idx) if outward else _get_boundary(lower_valley_idx,lower_peak_idx)
        upper_idx = _get_boundary(upper_peak_idx,upper_valley_idx) if outward else _get_boundary(upper_valley_idx,upper_peak_idx)

        return lower_idx, upper_idx
    
    def _fit_peaks(self,bins,counts,lower_idx,upper_idx,
                   fitter_class,initial_guess,n_gaussian=1,
                   poisson_statistic=False):
        '''
        Fit features and return the mean(s) and error(s) of the gaussian-approximated peak(s).

        Input:
            bins: 1D array, bin numbers
            counts: 1D array, counts
            lower_idx: int, lower boundary of the fitting region
            upper_idx: int, upper boundary of the fitting region
            fitter_class: class, fitter class
            initial_guess: 1D array, initial guess of the gaussian parameters
            n_gaussian: int, number of gaussian peaks
            poisson_statistic: bool, whether to consider poisson statistics

        Output:
            peak_mu: 1D array, mean of the gaussian-approximated peak(s)
            peak_mu_err: 1D array, error of the mean of the gaussian-approximated peak(s)
            peak_sigma: 1D array, standard deviation of the gaussian-approximated peak(s)
            peak_sigma_err: 1D array, error of the standard deviation of the gaussian-approximated peak(s)
            fitting_bins: 1D array, bin numbers of the fitting region
            fitted_counts: 1D array, counts of the gaussian-approximated peak(s)
        '''

        fitting_idx = np.arange(lower_idx,upper_idx+1)

        fitting_bins = bins[fitting_idx]

        count_normalizing_factor = 1 if poisson_statistic else np.sum(counts) * (bins[1] - bins[0])
        fitting_counts = counts[fitting_idx] / count_normalizing_factor
        fitting_counts_err = np.sqrt(counts)[fitting_idx] / count_normalizing_factor

        fitter = fitter_class(fitting_bins,fitting_counts,fitting_counts_err,initial_guess)
        fitting_result = fitter.fit()

        peak_mu = fitting_result['params'][:n_gaussian]
        peak_mu_err = fitting_result['e_params'][:n_gaussian] if fitting_result['e_params'] is not None else np.array([np.nan]*n_gaussian)

        peak_sigma = fitting_result['params'][n_gaussian:2*n_gaussian]
        peak_sigma_err = fitting_result['e_params'][n_gaussian:2*n_gaussian] if fitting_result['e_params'] is not None else np.array([np.nan]*n_gaussian)

        fitted_counts = fitter._get_model(fitting_bins,fitting_result['params']) * count_normalizing_factor
        return np.abs(peak_mu), peak_mu_err, peak_sigma, peak_sigma_err, fitting_bins, fitted_counts
    
    def _fit_single_peak(self,
                         bins,counts,kdes,
                         clear_peaks_idx,clear_valleys_idx,
                         approx_line_bin=None,threshold_ratio=1/3,
                         poisson_statistic=False,background_poly_order=0):
        '''
        Fit a single peak and return the mean and error of the gaussian-approximated peak if the approximated peak bin is provided.

        Input:
            bins: 1D array, bin numbers
            counts: 1D array, counts
            kdes: 1D array, smoothed density estimations
            clear_peaks_idx: 1D array, indices of the clear peaks
            clear_valleys_idx: 1D array, indices of the clear valleys
            approx_line_bin: int, approximate bin of the peak
            threshold_ratio: float, ratio of the threshold
            poisson_statistic: bool, whether to consider poisson statistics
            background_poly_order: int, order of the polynomial background

        Output:
            peak_mu: 1D array, mean of the gaussian-approximated peak
            peak_mu_err: 1D array, error of the mean of the gaussian-approximated peak
            peak_sigma: 1D array, standard deviation of the gaussian-approximated peak
            peak_sigma_err: 1D array, error of the standard deviation of the gaussian-approximated peak
            fitting_bins: 1D array, bin numbers of the fitting region
            fitted_counts: 1D array, counts of the gaussian-approximated peak
        '''

        if approx_line_bin is None:
            return np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([]), np.array([]), 

        # Find the fitting region
        approx_peak_idx = clear_peaks_idx[
            np.argmin(np.abs(bins[clear_peaks_idx] - approx_line_bin))
        ]
        approx_peak_bin = bins[approx_peak_idx]
        approx_peak_kde = kdes[approx_peak_idx]
        approx_peak_count = counts[approx_peak_idx]

        lower_idx, upper_idx = self._get_fitting_boundaries(bins,kdes,clear_peaks_idx,clear_valleys_idx,
                                                            approx_peak_bin,approx_peak_bin,
                                                            threshold=approx_peak_kde*threshold_ratio,
                                                            outward=True)
        
        # Get Gaussian initial guess
        mu_guess = approx_peak_bin
        sigma_guess = (bins[upper_idx] - bins[lower_idx]) / 2
        A_guess = approx_peak_kde * sigma_guess * np.sqrt(2*np.pi)
        if poisson_statistic:
            A_guess *= approx_peak_count / approx_peak_kde
        poly_params_guess = [0] * (background_poly_order + 1)

        initial_guess = [mu_guess,sigma_guess,A_guess,*poly_params_guess]

        # Fit
        fitter_class = GaussianPoissFitter if poisson_statistic else GaussianFitter
        return self._fit_peaks(bins,counts,lower_idx,upper_idx,fitter_class,initial_guess,poisson_statistic=poisson_statistic)
    
    def _get_peak_fwhm_counts(self,
                              bins,counts,count_time,
                              kdes,kernel_bw,
                              clear_peaks_idx,clear_valleys_idx,
                              peak_mu_bin,n_MC=1000):
        '''
        Get the counts within the FWHM of a peak.

        Input:
            bins: 1D array, bin numbers
            counts: 1D array, counts
            count_time: float, count time
            kdes: 1D array, smoothed density estimations
            kdes_err: 1D array, error of the smoothed density estimations
            clear_peaks_idx: 1D array, indices of the clear peaks
            clear_valleys_idx: 1D array, indices of the clear valleys
            peak_mu_bin: int, bin of the peak
            n_MC: int, number of Monte Carlo simulations

        Output:
            peak_fwhm_counts: float, counts within the FWHM of the peak
            peak_fwhm_counts_err: float, error of the counts within the FWHM of the peak
        '''

        # Find the fwhm region
        peak_mu_idx = clear_peaks_idx[np.argmin(np.abs(bins[clear_peaks_idx] - peak_mu_bin))]
        peak_mu_bin = bins[peak_mu_idx]
        peak_mu_kde = kdes[peak_mu_idx]

        fwhm_lower_idx, fwhm_upper_idx = self._get_fitting_boundaries(bins,kdes,clear_peaks_idx,clear_valleys_idx,
                                                                      peak_mu_bin,peak_mu_bin,
                                                                      threshold=peak_mu_kde/2,outward=True)

        threshold_err = (
            np.std(kdes[fwhm_lower_idx-kernel_bw:fwhm_lower_idx+kernel_bw+1]) + np.std(kdes[fwhm_upper_idx-kernel_bw:fwhm_upper_idx+kernel_bw+1])
        ) / 2
        # Get the counts within the fwhm region
        def _get_counts(threshold):
            lower_idx, upper_idx = self._get_fitting_boundaries(bins,kdes,clear_peaks_idx,clear_valleys_idx,
                                                                peak_mu_bin,peak_mu_bin,
                                                                threshold=threshold,outward=True)

            if lower_idx and upper_idx:
                fwhm_counts = np.sum(counts[lower_idx:upper_idx+1])
                return fwhm_counts, np.sqrt(fwhm_counts)

            return None, None

        if n_MC:
            peak_fwhm_counts = np.array([])
            peak_fwhm_counts_err = np.array([])

            for i in range(n_MC):
                threshold = np.random.normal(peak_mu_kde/2,threshold_err)
                fwhm_counts, fwhm_counts_err = _get_counts(threshold)

                if fwhm_counts is not None:
                    peak_fwhm_counts = np.append(peak_fwhm_counts,fwhm_counts)
                    peak_fwhm_counts_err = np.append(peak_fwhm_counts_err,fwhm_counts_err)

            peak_fwhm_rate = np.sum(peak_fwhm_counts/peak_fwhm_counts_err**2) / np.sum(1/peak_fwhm_counts_err**2) / count_time
            peak_fwhm_rate_err = np.sqrt(np.var(peak_fwhm_counts) + 1/np.sum(1/peak_fwhm_counts_err**2)) / count_time

            return peak_fwhm_rate, peak_fwhm_rate_err
        
        return _get_counts(fwhm_lower_idx, fwhm_upper_idx)


####################################################################################################################

'''
MCA Calibration Class
'''
class MCACalibration(MCAData):
    '''
    Calibration class for MCA data.
    '''

    def __init__(self,
                 na_22_data_file,ba_133_data_file,kernel_bw=5,
                 na_22_energy=511,na_22_approx_line_bin=1400,
                 ba_133_energies=[[53.15,79.60,81.00],[160.61],[276.40,302.85,356.01]],
                 ba_133_peak_ratios=[[0.3,0.3,1],[1],[0.2,0.4,1]],
                 cs_137_energy=661.7,cs_137_approx_line_bin=None,
                 non_calib_energies=[276.40]):
        
        self.na_22_energy = na_22_energy
        self.ba_133_energies = ba_133_energies
        self.cs_137_energy = cs_137_energy

        self.na_22_peak_mu, self.na_22_peak_mu_err, _, _, self.na_22_fitting_bins, self.na_22_fitted_counts = \
        self._fit_na_22_peak(na_22_data_file,kernel_bw,na_22_approx_line_bin)

        self.ba_133_peaks_mu, self.ba_133_peaks_mu_err, self.ba_133_fitting_bins, self.ba_133_fitted_counts, \
        self.cs_137_peak_mu, self.cs_137_peak_mu_err, self.cs_137_fitting_bins, self.cs_137_fitted_counts, self.cs_137_peak_sigma = \
        self._fit_ba_133_peaks(ba_133_data_file,kernel_bw,ba_133_energies,ba_133_peak_ratios,cs_137_approx_line_bin)

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
        '''
        Fit the peaks of Ba-133 and Cs-137 and return the mean and error of the gaussian-approximated peaks.

        Input:
            ba_133_data_file: string, path to the .Spe file of Ba-133
            ba_133_energies: list of lists, energies of the peaks of Ba-133
            ba_133_peak_ratios: list of lists, peak ratios of the peaks of Ba-133
            cs_137_approx_line_bin: int, approximate bin of the 661.7eV feature
            kernel_bw: float, bandwidth of the kernel

        Output:
            ba_133_peaks_mu: list of 1D arrays, means of the gaussian-approximated peaks
            ba_133_peaks_mu_err: list of 1D arrays, errors of the means of the gaussian-approximated peaks
            ba_133_fitting_bins: list of 1D arrays, bin numbers of the fitting regions
            ba_133_fitted_counts: list of 1D arrays, counts of the gaussian-approximated peaks
            cs_137_peak_mu: float, mean of the gaussian-approximated peak
            cs_137_peak_mu_err: float, error of the mean of the gaussian-approximated peak
            cs_137_fitting_bins: 1D array, bin numbers of the fitting region
            cs_137_fitted_counts: 1D array, counts of the gaussian-approximated peak
        '''

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

            lower_idx, upper_idx = self._get_fitting_boundaries(bins,kdes,clear_peaks_idx,clear_valleys_idx,
                                                                approx_peaks_idx[0],approx_peaks_idx[-1])

            n_gaussian = len(bundled_energies)
            # Get Gaussian initial guess
            mu_guess = approx_peaks_idx
            sigma_guess = (bins[upper_idx] - bins[lower_idx]) / 3 / n_gaussian
            A_guess = kdes[np.argmin(np.abs(bins - approx_peaks_idx[np.argmax(peak_ratios)]))] * sigma_guess * np.sqrt(2*np.pi) * np.array(peak_ratios)
            c_guess = 0

            initial_guess = [*mu_guess] + [sigma_guess]*n_gaussian + [*A_guess,c_guess]

            # Fit
            peaks_mu, peaks_mu_err, _, _, fitting_bins, fitted_counts = self._fit_peaks(bins,counts,lower_idx,upper_idx,
                                                                                        gaussian_fitter_classes[n_gaussian],
                                                                                        initial_guess,n_gaussian=n_gaussian)

            ba_133_peaks_mu.append(peaks_mu)
            ba_133_peaks_mu_err.append(peaks_mu_err)
            ba_133_fitting_bins.append(fitting_bins)
            ba_133_fitted_counts.append(fitted_counts)

        # Fit peaks of Cs-137 (if the approximate line is provided)
        cs_137_peak_mu, cs_137_peak_mu_err, cs_137_peak_sigma, _, \
        cs_137_fitting_bins, cs_137_fitted_counts = self._fit_single_peak(bins,counts,kdes,
                                                                          clear_peaks_idx,clear_valleys_idx,
                                                                          cs_137_approx_line_bin)

        return ba_133_peaks_mu, ba_133_peaks_mu_err, ba_133_fitting_bins, ba_133_fitted_counts, \
               cs_137_peak_mu, cs_137_peak_mu_err, cs_137_fitting_bins, cs_137_fitted_counts, cs_137_peak_sigma
    
    def _get_calib_stats(self,non_calib_energies):
        '''
        Get the calibration statistics.

        Input:
            non_calib_energies: list of floats, energies of the non-calibration peaks

        Attributes:
            calib_bins: 1D array, bin numbers of the calibration peaks
            calib_bins_err: 1D array, error of the bin numbers of the calibration peaks
            calib_energies: 1D array, energies of the calibration peaks
            energy_scaler: float, energy scaler
            energy_offset: float, energy offset
            energy_scaler_err: float, error of the energy scaler
            energy_offset_err: float, error of the energy offset
        '''

        # Get calibration statistics
        self.calib_bins = np.concatenate([self.na_22_peak_mu] + self.ba_133_peaks_mu + [self.cs_137_peak_mu])
        self.calib_bins_err = np.concatenate([self.na_22_peak_mu_err] + self.ba_133_peaks_mu_err + [self.cs_137_peak_mu_err])
        self.calib_energies = np.concatenate([np.array([self.na_22_energy])] + self.ba_133_energies + [np.array([self.cs_137_energy])])

        # Remove non-calibration peaks
        sorting_order = np.argsort(self.calib_energies)
        mask = np.ones(self.calib_bins.size,dtype=bool)
        for energy in non_calib_energies:
            mask[self.calib_energies[sorting_order] == energy] = False
        self.calib_bins = self.calib_bins[sorting_order][mask]
        self.calib_bins_err = self.calib_bins_err[sorting_order][mask]
        self.calib_energies = self.calib_energies[sorting_order][mask]

        # Calibrate the energy scaler and offset
        linear_fitter = NoUncertaintyLinearFitter(self.calib_bins,self.calib_energies,initial_guess=[np.mean(self.calib_energies/self.calib_bins),0])
        self.energy_scaler, self.energy_offset, self.energy_scaler_err, self.energy_offset_err = linear_fitter.fit()


'''
MCA Compton Scattering Class
'''
class MCACompton(MCAData):
    '''
    Compton scattering class for MCA data.
    '''
    def __init__(self,data_base='30_0304.Spe',data_dir='data/2025-03-04/',kernel_bw=5,):

        self.scatter_angle = np.deg2rad(float(data_base[:-9]))

        self.kernel_bw = kernel_bw

        self._read_data_and_calibrate(data_base,data_dir)

        for detector in ['recoil','scatter']:
            peak_mu_bin, peak_mu_bin_err, peak_sigma_bin, peak_sigma_bin_err, \
            peak_mu_energy, peak_mu_energy_err, peak_sigma_energy, peak_sigma_energy_err, \
            peak_fwhm_rate, peak_fwhm_rate_err, _, _, peak_mu_bin_deviation = self._peak_analysis(detector)

            self.__setattr__(detector+'_peak_mu_bin',peak_mu_bin)
            self.__setattr__(detector+'_peak_mu_bin_err',peak_mu_bin_err)
            self.__setattr__(detector+'_peak_sigma_bin',peak_sigma_bin)
            self.__setattr__(detector+'_peak_sigma_bin_err',peak_sigma_bin_err)
            self.__setattr__(detector+'_peak_mu_energy',peak_mu_energy)
            self.__setattr__(detector+'_peak_mu_energy_err',peak_mu_energy_err)
            self.__setattr__(detector+'_peak_sigma_energy',peak_sigma_energy)
            self.__setattr__(detector+'_peak_sigma_energy_err',peak_sigma_energy_err)
            self.__setattr__(detector+'_peak_fwhm_rate',peak_fwhm_rate)
            self.__setattr__(detector+'_peak_fwhm_rate_err',peak_fwhm_rate_err)
            self.__setattr__(detector+'_peak_mu_bin_deviation',peak_mu_bin_deviation)

    def _read_data_and_calibrate(self,data_base,data_dir):
        '''
        Read data and calibration files and calibrate the energy scaler.

        Input:
            data_base: string, base name of the data files
            data_dir: string, path to the data directory

        Attributes:
            for each of "recoil" and "scatter" detectors:
                (detector)_data_file: string, path to the data file
                (detector)_na_calibration_data_file: string, path to the Na-22 calibration data file
                (detector)_ba_calibration_data_file: string, path to the Ba-133 calibration data file
                (detector)_energy_scaler: float, energy scaler
                (detector)_energy_offset: float, energy offset
                (detector)_energy_scaler_err: float, error of the energy scaler
                (detector)_energy_offset_err: float, error of the energy offset
        '''

        for detector in ['recoil','scatter']:
            # Get data and calibration files
            self.__setattr__(detector+'_data_file',os.path.join(data_dir,detector+'_'+data_base))
            for calib_type in ['na','ba']:
                self.__setattr__(detector+'_'+calib_type+'_calibration_data_file',os.path.join(data_dir,f'Calibration_{detector}_{calib_type}'+data_base[-9:]))
            
            # Calibrate the energy scaler
            calibration = MCACalibration(
                na_22_data_file = self.__getattribute__(f'{detector}_na_calibration_data_file'),
                ba_133_data_file = self.__getattribute__(f'{detector}_ba_calibration_data_file'),
                kernel_bw = self.kernel_bw,cs_137_approx_line_bin=1800 if detector == 'recoil' else None
            )
            self.__setattr__(detector+'_energy_scaler',calibration.energy_scaler)
            self.__setattr__(detector+'_energy_offset',calibration.energy_offset)
            self.__setattr__(detector+'_energy_scaler_err',calibration.energy_scaler_err)
            self.__setattr__(detector+'_energy_offset_err',calibration.energy_offset_err)

            if detector == 'recoil':
                self.cs_137_peak_width = (calibration.cs_137_peak_sigma / (calibration.cs_137_peak_mu + calibration.energy_offset/calibration.energy_scaler))[0] * 2
    
    def _bin_to_energy(self,bins,bins_err,detector):
        '''
        Convert bins to energies.

        Input:
            bins: 1D array, bin numbers
            bins_err: 1D array, error of the bin numbers
            detector: string, detector type

        Output:
            energy: 1D array, energies
            energy_err: 1D array, error of the energies
        '''

        energy_scaler = self.__getattribute__(f'{detector}_energy_scaler')
        energy_offset = self.__getattribute__(f'{detector}_energy_offset')
        energy_scaler_err = self.__getattribute__(f'{detector}_energy_scaler_err')
        energy_offset_err = self.__getattribute__(f'{detector}_energy_offset_err')

        energy = bins * energy_scaler + energy_offset
        energy_err = np.sqrt(
            ((bins_err/bins)**2 + (energy_scaler_err/energy_scaler)**2) * (bins*energy_scaler)**2 + energy_offset_err**2
        )

        return energy, energy_err

    def _peak_analysis(self,detector):
        '''
        Analyze the peaks of the Compton scattering features.

        Input:
            detector: string, detector type

        Output:
            peak_mu_count: float, mean of the gaussian-approximated peak
            peak_mu_count_err: float, error of the mean of the gaussian-approximated peak
            peak_sigma_bin: float, standard deviation of the gaussian-approximated peak
            peak_sigma_bin_err: float, error of the standard deviation of the gaussian-approximated peak
            peak_mu_energy: float, mean of the gaussian-approximated peak in energy
            peak_mu_energy_err: float, error of the mean of the gaussian-approximated peak in energy
            peak_sigma_energy: float, standard deviation of the gaussian-approximated peak in energy
            peak_sigma_energy_err: float, error of the standard deviation of the gaussian-approximated peak in energy
            peak_fwhm_rate: float, FWHM of the peak in count per second
            peak_fwhm_rate_err: float, error of the FWHM of the peak in count per second
            fitting_bins: 1D array, bin numbers of the fitting region
            fitted_counts: 1D array, counts of the gaussian-approximated peak
            peak_mu_bin_deviation: float, deviation of the mean of the gaussian-approximated peak from the feature bin
        '''

        data_file = self.__getattribute__(f'{detector}_data_file')

        # Read data
        bins, counts, count_time, _ = self._read_data(data_file)
        bins, kdes, kdes_err = self._kde_smooth_data(bins,counts,self.kernel_bw)

        # Find peaks and valleys
        clear_peaks_idx, clear_valleys_idx = self._find_peaks_and_valleys(bins,kdes,kdes_err)
        
        # Fit peak
        peak_mu_bin, peak_mu_bin_err, peak_sigma_bin, peak_sigma_bin_err, fitting_bins, fitted_counts,  = \
        self._fit_single_peak(bins,counts,kdes,clear_peaks_idx,clear_valleys_idx,
                              bins[clear_peaks_idx[0]],threshold_ratio=1/2,
                              poisson_statistic=True,background_poly_order=-1)
        
        peak_mu_bin_deviation = peak_mu_bin - bins[clear_peaks_idx[0]]
        
        peak_mu_energy, peak_mu_energy_err = self._bin_to_energy(peak_mu_bin,peak_mu_bin_err,detector)
        peak_sigma_energy, peak_sigma_energy_err = self._bin_to_energy(peak_sigma_bin,peak_sigma_bin_err,detector)

        # Get FWHM
        peak_fwhm_rate, peak_fwhm_rate_err = self._get_peak_fwhm_counts(bins,counts,count_time,
                                                                        kdes,self.kernel_bw,
                                                                        clear_peaks_idx,clear_valleys_idx,
                                                                        peak_mu_bin,n_MC=1000)

        return peak_mu_bin, peak_mu_bin_err, peak_sigma_bin, peak_sigma_bin_err, \
               peak_mu_energy, peak_mu_energy_err, peak_sigma_energy, peak_sigma_energy_err, \
               peak_fwhm_rate, peak_fwhm_rate_err, fitting_bins, fitted_counts, peak_mu_bin_deviation