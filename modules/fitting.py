import numpy as np

from scipy.optimize import minimize, least_squares
from scipy.stats import chi2

class BaseFitter:
    def __init__(self, x, y, yerr):
        self.x = x
        self.y = y
        self.yerr = yerr

    def _get_model(self, x, params):
        raise NotImplementedError("Not implemented in base class.")
    
    def _get_residuals(self, params):
        yhat = self._get_model(self.x,params)
        return (self.y - yhat)**2 / self.yerr**2
    
    def _get_chisqr(self, params):
        return np.sum(self._get_residuals(params))
    
    def _get_initial_guess(self):
        raise NotImplementedError("Not implemented in base class.")
    
    def fit(self):
        initial_guess = self._get_initial_guess()
        result = least_squares(
            self._get_residuals, initial_guess
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
            'alpha': alpha,

            'cov': cov,

            'success': result.success,
            'message': result.message,
        }

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