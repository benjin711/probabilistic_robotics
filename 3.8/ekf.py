from dataclasses import dataclass
from typing import Callable

import numpy as np

from kf import Gaussian


@dataclass
class EKFParams:
    R: np.ndarray
    Q: np.ndarray
    g: Callable
    h: Callable
    G: Callable
    H: Callable


class EKF:
    def __init__(
        self,
        params: EKFParams,
    ):
        self.R = params.R  
        self.Q = params.Q
        self.g = params.g
        self.h = params.h
        self.G = params.G
        self.H = params.H

    def prediction_update(self, bel: Gaussian, u: np.ndarray) -> Gaussian:
        mu = self.g(bel.mu, u)

        G = self.G(bel.mu, u)
        Sigma = G @ bel.Sigma @ G.T
        if self.R is not None:
            Sigma += self.R
        
        return Gaussian(mu, Sigma)
    
    def measurement_update(self, bel: Gaussian, z: np.ndarray) -> Gaussian:
        if self.h is None or self.H is None or z is None:
            return bel
        
        H = self.H(bel.mu)
        K = bel.Sigma @ H.T
        if self.Q is not None:
            K = K @ np.linalg.inv(H @ bel.Sigma @ H.T + self.Q)
        else:
            K = K @ np.linalg.inv(H @ bel.Sigma @ H.T)

        mu = bel.mu + K @ (z - self.h(bel.mu))
        Sigma = (np.eye(bel.mu.shape[0]) - K @ H) @ bel.Sigma
        return Gaussian(mu, Sigma)

    def update(self, bel: Gaussian, u: np.ndarray = None, z: np.ndarray = None) -> Gaussian:
        pred_bel = self.prediction_update(bel, u)
        meas_bel = self.measurement_update(pred_bel, z) 
        return meas_bel, pred_bel