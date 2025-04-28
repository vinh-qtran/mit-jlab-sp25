import numpy as np

from scipy.interpolate import RegularGridInterpolator

from tqdm import tqdm

import json

#################################################################################

class BaseDetectorSimulation:
    def __init__(self, 
                 d, h, w,
                 n_theta = 100, n_phi = 100,
                 n_points=10000):
        
        self.d = d  # Detector distance
        self.h = h  # Detector height
        self.w = w  # Detector width

        self.base_thetas = np.pi/2 - np.linspace(0, np.arctan(self.h / np.sqrt(self.d**2 + self.w**2)), n_theta)
        self.base_phis = np.pi/2 - np.linspace(0, np.arctan(self.d / self.w), n_phi)

        base_points = self._generate_base_points(n_points)
        
        self.base_differential_cross_sections = np.zeros((n_theta, n_phi))

        for i in tqdm(range(n_theta)):
            for j in range(n_phi):
                theta = self.base_thetas[i]
                phi = self.base_phis[j]

                self.base_differential_cross_sections[i, j] = self._get_differential_cross_section(base_points, theta, phi)

    def _generate_base_points(self, n_points):
        x = np.random.uniform(-self.w / 2, self.w / 2, n_points)
        y = np.zeros(n_points)
        z = np.random.uniform(-self.h / 2, self.h / 2, n_points)

        return np.column_stack((x, y, z))
    
    def _get_hit_counts(self, base_points, theta, phi):
        delta_x = self.d / np.tan(phi) if phi > 0 else 0
        delta_z = self.d / np.sin(phi) / np.tan(theta) if theta > 0 else 0

        hit_points = base_points + np.array([delta_x, self.d, delta_z])

        return np.sum(np.logical_and(
            np.logical_and(
                hit_points[:, 0] >= -self.w / 2,
                hit_points[:, 0] <= self.w / 2
            ),
            np.logical_and(
                hit_points[:, 2] >= -self.h / 2,
                hit_points[:, 2] <= self.h / 2
            )
        ))
    
    def _get_differential_cross_section(self, base_points, theta, phi):
        hit_counts = self._get_hit_counts(base_points, theta, phi)

        cross_section_scaler = self.h * self.w * np.sin(theta) * np.sin(phi)

        return hit_counts / base_points.shape[0] * cross_section_scaler
    
class DetectorCrossSection():
    def __init__(self, 
                 theta_de, detector_simulation,
                 thetas = np.linspace(0, np.pi, 1801),
                 phis = np.linspace(-np.pi/2, 3*np.pi/2, 1801),):
        self._get_differential_cross_section = RegularGridInterpolator(
            (
                np.pi/2 - detector_simulation.base_thetas,
                np.pi/2 - detector_simulation.base_phis
            ),
            detector_simulation.base_differential_cross_sections, 
            bounds_error=False, fill_value=0
        )

        self.theta_de = theta_de
        self.thetas = thetas
        self.phis = phis

        self.phis_grid, self.thetas_grid = np.meshgrid(self.phis, self.thetas)

        delta_thetas_prime_grid, delta_phis_prime_grid = self._get_delta_prime_coordinates(self.thetas_grid.flatten(), self.phis_grid.flatten(), np.pi/2-self.theta_de)

        self.differential_cross_sections = self._get_differential_cross_section(
            (delta_thetas_prime_grid.flatten(), delta_phis_prime_grid.flatten())
        ).reshape(self.thetas_grid.shape)

    def _coordinate_transform(self, thetas, phis, alpha):
        thetas_prime = np.arccos(
            - np.sin(alpha)*np.sin(thetas)*np.sin(phis) + np.cos(alpha)*np.cos(thetas)
        )
        phis_prime = np.arctan2(
            np.cos(alpha)*np.sin(thetas)*np.sin(phis) + np.sin(alpha)*np.cos(thetas),
            np.sin(thetas)*np.cos(phis)
        )
        return thetas_prime, phis_prime
    
    def _get_delta_prime_coordinates(self, thetas, phis, alpha):
        thetas_prime, phis_prime = self._coordinate_transform(thetas, phis, alpha)

        delta_thetas_prime = np.abs(np.pi/2 - thetas_prime)
        delta_phis_prime = np.abs(np.pi/2 - phis_prime)

        return delta_thetas_prime, delta_phis_prime

    
if __name__ == "__main__":
    detector_simulation = BaseDetectorSimulation(
        d=6.5,
        h=1.0,
        w=5.0,
        n_theta=1000,
        n_phi=500,
        n_points=40000
    )

    for theta_de in tqdm(np.linspace(0,90,10)):
        detector_cross_section = DetectorCrossSection(
            theta_de=theta_de*np.pi/180,
            detector_simulation=detector_simulation,
        )

        with open(f'../angle_analysis/detector_cross_sections/{theta_de}.json', 'w') as f:
            json.dump({
                "theta": detector_cross_section.thetas.tolist(),
                "phi": detector_cross_section.phis.tolist(),
                "dsigma_dOmega": detector_cross_section.differential_cross_sections.tolist()
            }, f)