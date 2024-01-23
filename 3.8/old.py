from typing import List, Tuple, Union
from matplotlib import transforms
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

@dataclass
class MVGaussian:
    mu: np.ndarray
    cov: np.ndarray

    def __repr__(self):
        return f"MVGaussian(\n\tmu={self.mu},\n\tcov={self.cov}\n)"
    
def confidence_ellipse(mu, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    mu : ndarray, shape (2,)
        The mean value.

    cov : np.ndarray
        The 2x2 covariance matrix to base the ellipse on.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mu[0], mu[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    
def visualize(progression: List[MVGaussian]):
    _, ax = plt.subplots()

    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 2)

    for state in progression:
        ax.scatter(state.mu[0], state.mu[1], c='b', marker='x')
        confidence_ellipse(state.mu, state.cov, ax, n_std=1, edgecolor='b', facecolor='b', alpha=0.1)

    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.show()

def visualize2(progression1: List[MVGaussian], progression2: List[MVGaussian]):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1x2 subplot

    # Plot for progression1
    axs[0].set_xlim(-15, 15)
    axs[0].set_ylim(-5, 5)
    for state in progression1:
        axs[0].scatter(state.mu[0], state.mu[1], c='b', marker='x')
        confidence_ellipse(state.mu, state.cov, axs[0], n_std=1, edgecolor='b', facecolor='b', alpha=0.1)
    axs[0].set_xlabel('Position')
    axs[0].set_ylabel('Velocity')

    # Plot for progression2
    axs[1].set_xlim(-15, 15)
    axs[1].set_ylim(-5, 5)
    for state in progression2:
        axs[1].scatter(state.mu[0], state.mu[1], c='r', marker='x')
        confidence_ellipse(state.mu, state.cov, axs[1], n_std=1, edgecolor='r', facecolor='r', alpha=0.1)
    axs[1].set_xlabel('Position')
    axs[1].set_ylabel('Velocity')

    plt.tight_layout()
    plt.show()

def KF(state: MVGaussian, measurement: np.ndarray = None) -> MVGaussian:
    A = np.array([
        [1, 1],
        [0, 1]
    ])
    R = np.array([
        [1/4, 1/4],
        [1/4, 1/4]
    ])

    # Prediction step
    mu_bar = A @ state.mu
    cov_bar = A @ state.cov @ A.T + R

    # Measurement step
    C = np.array([
        [1, 0]
    ])
    Q = np.array([
        10
    ])

    if measurement is not None:
        K = cov_bar @ C.T * (1 / (C @ cov_bar @ C.T + Q))
        mu_bar = mu_bar + K @ (measurement - C @ mu_bar)
        cov_bar = (np.eye(2) - K @ C) @ cov_bar

    return MVGaussian(
        mu=mu_bar,
        cov=cov_bar
    )

def correlation_coef(cov: np.ndarray) -> float:
    return cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

def run_progression(iters: int, measurements: List[Union[int, None]]) -> Tuple[List[MVGaussian], List[float]]:
    progression = []
    correlation = []

    state1 = MVGaussian(
        mu=np.array([0, 0]),
        cov=np.array([
            [1/4, 1/4],
            [1/4, 1/4]
        ])
    )

    progression.append(state1)
    correlation.append(correlation_coef(state1.cov))

    for i in range(iters):
        state = KF(progression[-1], measurements[i])
        progression.append(state)
        correlation.append(correlation_coef(state.cov))

    return progression, correlation
    
if __name__ == "__main__":
    N = 10

    measurements1 = [None] * N
    prog1, corr1 = run_progression(N, measurements1)

    measurements2 = [None] * N
    measurements2[4] = 5
    prog2, corr2 = run_progression(N, measurements2)

    visualize2(prog1, prog2)
    
    # visualize(progression)
    # plt.plot(correlation)
    # plt.show()
