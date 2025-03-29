import numpy as np

from scipy.optimize import minimize, least_squares
from scipy.stats import poisson, chi2, f

from matplotlib import pyplot as plt

from tqdm import tqdm

# BASE FITTERS

class BaseFitter:
    def __init__(self, x, y, yerr, xerr=None):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.xerr = xerr

        if self.xerr is not None:
            self.dx = (self.x.max() - self.x.min()) / 1000

    def _normalize(self,x):
        return x.mean(), x.std()

    def _get_model(self, x, params):
        raise NotImplementedError("Not implemented in base class.")
    
    def _get_residuals(self, params):
        yhat = self._get_model(self.x,params)
        if self.xerr is not None:
            dyhat_dx = (self._get_model(self.x + self.dx,params) - self._get_model(self.x - self.dx,params)) / (2*self.dx)
            
            sigma_sqr = self.yerr**2 + dyhat_dx**2 * self.xerr**2
        else:
            sigma_sqr = self.yerr**2
        
        return (self.y - yhat)**2 / sigma_sqr
    
    def _get_chisqr(self, params):
        return np.sum(self._get_residuals(params))
    
    def _get_initial_guess(self):
        raise NotImplementedError("Not implemented in base class.")
    
    def fit(self,bounds=(-np.inf,np.inf)):
        initial_guess = self._get_initial_guess()
        result = least_squares(
            self._get_residuals, initial_guess, bounds=bounds
        )

        chisqr = self._get_chisqr(result.x)
        alpha = 1 - chi2.cdf(chisqr, len(self.x) - len(result.x))

        try:
            cov = np.linalg.inv(np.dot(result.jac.T, result.jac))
            e_params = np.sqrt(np.diagonal(cov))
        except np.linalg.LinAlgError:
            cov = None
            e_params = None

        return {
            'params': result.x,
            'e_params': e_params,

            'chisqr': self._get_chisqr(result.x),
            'reduced_chisqr': self._get_chisqr(result.x) / (len(self.x) - len(result.x)),
            'alpha': alpha,

            'cov': cov,

            'success': result.success,
            'message': result.message,
        }
    
    def model_interpolation(self, x=None, params=None, extend=0.2):
        if x is None:
            x_range = self.x.max() - self.x.min()
            x = np.linspace(self.x.min() - extend*x_range, 
                            self.x.max() + extend*x_range,
                            1000)

        if params is None:
            params = self.fit()['params']

        return x, self._get_model(x, params)
    
class BasePoissonFitter(BaseFitter):
    def __init__(self, x, y):
        mask = y > 0

        self.x = x[mask]
        self.y = y[mask]

    def _get_residuals(self, params):
        yhat = self._get_model(self.x,params)

        C = np.log(2*np.pi*self.y) if (yhat == 0).any() else np.log(2*np.pi*yhat)

        return - 2*poisson.logpmf(self.y, yhat) - C
    
class BaseUniformMonteCarloFitter():
    def __init__(self,
                 x,x_err,
                 y,y_err,
                 fitter:BaseFitter,
                 N_sample=10000):
        self.x = x
        self.x_err = x_err

        self.y = y
        self.y_err = y_err

        self.fitter = fitter

        self.N_sample = N_sample

    def _sample_fit(self,sampled_x):
        fitter = self.fitter(sampled_x, self.y, self.y_err)
        fitting_result = fitter.fit()
        return fitting_result['params'], fitting_result['e_params']
    
    def fit(self):
        params = []
        e_params = []

        for i in tqdm(range(self.N_sample)):
            sampled_x = np.random.uniform(self.x - self.x_err,self.x + self.x_err)
            sampled_params, sampled_e_params = self._sample_fit(sampled_x)

            if sampled_e_params is None or np.any(np.isnan(sampled_e_params)):
                continue

            params.append(sampled_params)
            e_params.append(sampled_e_params)

        params = np.concatenate(params).reshape(-1,len(sampled_params))
        e_params = np.concatenate(e_params).reshape(-1,len(sampled_params))

        weights = 1 / e_params**2
        params_mean = np.sum(params * weights, axis=0) / np.sum(weights, axis=0)
        params_std = np.sqrt(np.var(params, axis=0) + 1 / np.sum(weights, axis=0))

        return params_mean, params_std, params, e_params

# SPECIFIC FITTERS

class BayesianGaussian:
    def __init__(self,x):
        self.x = x

        params = self._fit()
        self.mu, self.sigma = params
        self.chisqr, self.alpha = self._get_chisqr_stats(params)

    def _get_NLL(self, params):
        mu, sigma = params
        return np.sum(
            np.log(np.sqrt(2*np.pi*sigma**2)) + (self.x - mu)**2 / (2*sigma**2)
        )

    def _get_initial_guess(self):
        return [np.mean(self.x), np.std(self.x)]
    
    def _fit(self):
        initial_guess = self._get_initial_guess()
        result = minimize(
            self._get_NLL, initial_guess
        )
        return result.x
    
    def _get_chisqr_stats(self, params):
        mu, sigma = params
        chisqr = np.sum((self.x - mu)**2 / sigma**2)
        alpha = 1 - chi2.cdf(chisqr, len(self.x) - len(params))
        return chisqr, alpha


class PolynomialFitter(BaseFitter):
    def __init__(self, 
                 x, y, yerr,
                 order=0,
                 initial_guess=None):
        super().__init__(x, y, yerr)

        self.order = order
        self.initial_guess = initial_guess

        if self.initial_guess:
            assert len(self.initial_guess) == self.order + 1, "Initial guess does not match polynomial order."
        else:
            self.initial_guess = np.zeros(self.order + 1)

    def _get_model(self, x, params):
        return np.polyval(params, x)
    
    def _get_initial_guess(self):
        return self.initial_guess
    
class ChowPolynomial:
    def __init__(self, 
                 x, y, yerr,
                 max_order=7,
                 alpha=0.05):
        self.x = x
        self.y = y
        self.yerr = yerr

        self.n = len(x)

        self.max_order = max_order
        self.alpha = alpha

    def _get_model(self, x, params):
        return np.polyval(params, x)
    
    def _get_model_derivative(self, x, params, order=1):
        return np.polyval(np.polyder(params, order), x)

    def _fit_polynomial(self, order):
        fitter = PolynomialFitter(
            self.x, self.y, self.yerr,
            order=order
        )
        return fitter.fit()
    
    def fit(self):
        param_nums = np.arange(1, self.max_order + 1)
        fitting_results = [
            self._fit_polynomial(param_num-1) for param_num in param_nums
        ]

        F_ratio = []

        for i in range(self.max_order-1):
            F_stat = (
                (fitting_results[i]['chisqr'] - fitting_results[i+1]['chisqr']) /
                (fitting_results[i+1]['chisqr'] / (self.n - param_nums[i+1]))
            )

            F_crit = f.ppf(1 - self.alpha, 1, self.n - param_nums[i+1])

            F_ratio.append(F_stat / F_crit)

            if F_stat < F_crit:
                return fitting_results[i]
            
        return fitting_results[-1]
    

# ANALYSIS TOOLS

class NSigma:
    def __init__(self,params,cov,par0_idx=0,par1_idx=1,range=3,resolution=100):
        if cov is None:
            raise ValueError("Covariance matrix is not available.")
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance matrix is not square.")
        if cov.shape[0] != len(params):
            raise ValueError("Covariance matrix does not match parameter length.")
        
        mask = [par0_idx,par1_idx]

        self.cov_inv = np.linalg.inv(cov)[mask][:,mask]

        self.params = params[mask]
        self.e_params = np.sqrt(np.diagonal(np.linalg.inv(self.cov_inv)))

        self.range = range
        self.resolution = resolution

    def _extend_param(self,param,e_param):
        return np.linspace(-self.range,self.range,self.resolution) * e_param + param
    
    def _get_par_and_delta_par(self):        
        par0 = self._extend_param(self.params[0],self.e_params[0])
        par1 = self._extend_param(self.params[1],self.e_params[1])

        delta_par0 = par0 - self.params[0]
        delta_par1 = par1 - self.params[1]

        return par0, par1, delta_par0, delta_par1
    
    def _get_n_sigma(self,i_delta_par0,i_delta_par1):
        sig_squared = np.dot(np.dot([i_delta_par0, i_delta_par1], self.cov_inv),
                             [i_delta_par0, i_delta_par1])
        return np.sqrt(sig_squared)
    
    def get_n_sigma_grids(self):
        par0, par1, delta_par0, delta_par1 = self._get_par_and_delta_par()

        par0_grid, par1_grid = np.meshgrid(par0,par1)
        n_sigma_grid = np.array([
            [self._get_n_sigma(i_delta_par0,i_delta_par1) for i_delta_par0 in delta_par0]
            for i_delta_par1 in delta_par1
        ])

        return par0_grid, par1_grid, n_sigma_grid
    
    def show_n_sigma(self,fig,ax,par0_scaler=1,par1_scaler=1,cb_loc='top'):
        par0_grid, par1_grid, n_sigma_grid = self.get_n_sigma_grids()
        par0_grid *= par0_scaler
        par1_grid *= par1_scaler

        c = ax.pcolor(par0_grid,par1_grid,n_sigma_grid,cmap='RdBu',vmin=0,vmax=5)
        con = ax.contour(par0_grid,par1_grid,n_sigma_grid,levels=[1,2,3],colors='black',linewidths=2.5, linestyles=':', alpha=1)
        plt.clabel(con, inline=True, fmt=r'$%.0f \, \sigma$', fontsize=24, use_clabeltext=True)

        cb = fig.colorbar(c, ax=ax, ticks=[0,1,2,3,4,5], location=cb_loc)
        cb.set_label(r'$N \sigma$')

        ax.set_xlim(par0_grid.min(),par0_grid.max())
        ax.set_ylim(par1_grid.min(),par1_grid.max())