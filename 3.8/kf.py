import numpy as np
from dataclasses import dataclass

@dataclass
class Gaussian:
    mu: np.ndarray
    Sigma: np.ndarray

    def __repr__(self):
        return f"Gaussian(\nmu=\n{self.mu},\ncov=\n{self.Sigma}\n)"

@dataclass
class KFParams:
    A: np.ndarray
    B: np.ndarray
    R: np.ndarray
    C: np.ndarray
    Q: np.ndarray

class KF:
    def __init__(
        self, 
        params: KFParams
    ):
        self.A = params.A
        self.B = params.B
        self.R = params.R
        self.C = params.C
        self.Q = params.Q

    def prediction_update(self, bel: Gaussian, u: np.ndarray) -> Gaussian:
        mu = self.A @ bel.mu
        if self.B is not None and u is not None:
            mu += self.B @ u

        Sigma = self.A @ bel.Sigma @ self.A.T
        if self.R is not None:
            Sigma += self.R
        
        return Gaussian(mu, Sigma)
    
    def measurement_update(self, bel: Gaussian, z: np.ndarray) -> Gaussian:
        if self.C is None:
            return bel
        
        K = bel.Sigma @ self.C.T
        if self.Q is not None:
            K = K @ np.linalg.inv(self.C @ bel.Sigma @ self.C.T + self.Q)
        else:
            K = K @ np.linalg.inv(self.C @ bel.Sigma @ self.C.T)
        
        mu = bel.mu + K @ (z - self.C @ bel.mu)
        Sigma = (np.eye(bel.mu.shape[0]) - K @ self.C) @ bel.Sigma
        return Gaussian(mu, Sigma)

    def update(self, bel: Gaussian, u: np.ndarray = None, z: np.ndarray = None) -> Gaussian:
        pred_bel = self.prediction_update(bel, u)
        meas_bel = self.measurement_update(pred_bel, z) 
        return meas_bel, pred_bel

