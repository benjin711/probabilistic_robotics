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
        G = self.G(bel.mu, u)
        return Gaussian(
            self.g(bel.mu, u),
            G @ bel.Sigma @ G + self.R
        )
    
    def measurement_update(self, bel: Gaussian, z: np.ndarray) -> Gaussian:
        H = self.H(bel.mu)
        K = bel.Sigma @ H.T @ np.linalg.inv(H @ bel.Sigma @ H.T + self.Q)
        return Gaussian(
            bel.mu + K @ (z - self.h(bel.mu)),
            (np.eye(bel.mu.shape[0]) - K @ H) @ bel.Sigma
        )

    def update(self, bel: Gaussian, u: np.ndarray, z: np.ndarray) -> Gaussian:
        pred_bel = self.prediction_update(bel, u)
        meas_bel = self.measurement_update(pred_bel, z) 
        return meas_bel, pred_bel