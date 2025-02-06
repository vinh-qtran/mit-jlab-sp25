import numpy as np

from scipy.optimize import minimize, least_squares
from scipy.stats import chi2

class BaseFitter:
    def __init__(self, x, y, yerr):
        self.x = x
        self.y = y
        self.yerr = yerr

    def _get_model(self, params):
        raise NotImplementedError("Not implemented in base class.")
    
    def _get_residuals(self, params):
        yhat = self._get_model(params)
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

        try:
            cov = np.linalg.inv(np.dot(result.jac.T,result.jac))
            e_params = np.sqrt(np.diagonal(cov))
        except np.linalg.LinAlgError:
            cov = None
            e_params = None

        return {
            'params': result.x,
            'e_params': e_params,

            'chisqr': self._get_chisqr(result.x),
            'cov': cov,

            'success': result.success,
            'message': result.message,
        }
