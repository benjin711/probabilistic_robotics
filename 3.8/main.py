import sys
from matplotlib import pyplot as plt, transforms
from matplotlib.patches import Ellipse
from scipy.stats import chi2

import numpy as np
from kf import KF, Gaussian, KFParams

def parse_args():
    # sys.argv[0] is the script name itself
    if len(sys.argv) == 2:
        first_arg = sys.argv[1]
        return first_arg
    
    raise ValueError("Invalid arguments. Please provide a single argument corresponding to the task to run, e.g. 1c.")

def filter_config_v1():
    A = np.array([
        [1, 1],
        [0, 1]
    ], dtype=np.float64)
    R = np.array([
        [1/4, 1/2],
        [1/2, 1]
    ], dtype=np.float64)
    return KFParams(A, None, R, None, None)

def filter_config_v2():
    A = np.array([
        [1, 1],
        [0, 1]
    ], dtype=np.float64)
    R = np.array([
        [1/4, 1/2],
        [1/2, 1]
    ], dtype=np.float64)
    C = np.array([
        [1, 0]
    ], dtype=np.float64)
    Q = np.array([
        [10]
    ], dtype=np.float64)
    return KFParams(A, None, R, C, Q)

def init_state():
    return Gaussian(
        mu=np.array([0, 0], dtype=np.float64),
        Sigma=np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.float64)
    )

def draw_uncertainty_ellipse(mu, Sigma, ax, p=0.68, **kwargs):
    """
    Draw an uncertainty ellipse at the given confidence level. p = 0.68 means
    that the ellipse will contain 68% of the probability mass. Theoretical 
    justification for this can be found at 
    https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix
    """
    # Get the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(Sigma)

    value = chi2.ppf(1 - p, 2)
    width = 2 * np.sqrt(value * eigvals[0])
    height = 2 * np.sqrt(value * eigvals[1])

    # Make width the larger value
    if width < height:
        width, height = height, width
        eigvecs = eigvecs[:, [1, 0]]

    # Get the angle of rotation
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    # Draw the ellipse
    ellipse = Ellipse(xy=mu, width=width, height=height, angle=np.degrees(angle), **kwargs)
    ellipse.set_transform(ax.transData)

    ax.add_patch(ellipse)

def f1c():
    x = init_state()
    filter = KF(filter_config_v1())

    ITERS = 5

    print(x)
    for _ in range(ITERS):
        x, _ = filter.update(x, None, None)
        print(x)

def f1d():
    x = init_state()
    filter = KF(filter_config_v1())

    ITERS = 5

    _, ax = plt.subplots()
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2, 2)

    ax.scatter(x.mu[0], x.mu[1], c='b', marker='x')
    draw_uncertainty_ellipse(x.mu, x.Sigma, ax, p=0.68, edgecolor='b', facecolor='b', alpha=0.1)

    for _ in range(ITERS):
        x, _ = filter.update(x, None, None)
        ax.scatter(x.mu[0], x.mu[1], c='b', marker='x')
        draw_uncertainty_ellipse(x.mu, x.Sigma, ax, p=0.68, edgecolor='b', facecolor='b', alpha=0.1)

    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.savefig('3.8/1d.png')

def f1e():
    x = init_state()
    filter = KF(filter_config_v1())

    ITERS = 10

    pccs = [float("nan")]

    for _ in range(ITERS):
        x, _ = filter.update(x, None, None)
        pccs.append(x.Sigma[0, 1] / np.sqrt(x.Sigma[0, 0] * x.Sigma[1, 1]))

    plt.plot(pccs)
    plt.xlabel('timestep')
    plt.ylabel('Pearson correlation coefficient')
    plt.savefig('3.8/1e.png')

def f2b():
    x = init_state()
    filter = KF(filter_config_v2())

    ITERS = 5

    beliefs = [(x, x)]
    measurements = [None, None, None, None, 5]

    _, ax = plt.subplots()
    ax.set_xlim(-6, 7)
    ax.set_ylim(-2, 3)

    ax.scatter(x.mu[0], x.mu[1], c='b', marker='x')
    draw_uncertainty_ellipse(x.mu, x.Sigma, ax, p=0.68, edgecolor='b', facecolor='b', alpha=0.1)

    for idx in range(ITERS):
        x, x_pre = filter.update(x, None, measurements[idx])
        beliefs.append((x, x_pre))
        ax.scatter(x_pre.mu[0], x_pre.mu[1], c='b', marker='x')
        draw_uncertainty_ellipse(x_pre.mu, x_pre.Sigma, ax, p=0.68, edgecolor='b', facecolor='b', alpha=0.1)

    for measurement in measurements:
        if measurement is not None:
            ax.axvline(x=measurement, color='g', alpha=0.5)

    ax.scatter(x.mu[0], x.mu[1], c='b', marker='x')
    draw_uncertainty_ellipse(x.mu, x.Sigma, ax, p=0.68, edgecolor='b', facecolor='g', alpha=0.1)

    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.savefig('3.8/2b.png')

    print(beliefs[-1][1])
    print(beliefs[-1][0])

def main():
    task = parse_args()
    if task == "1c":
        f1c()
    elif task == "1d":
        f1d()
    elif task == "1e":
        f1e()
    elif task == "2b":
        f2b()

if __name__ == "__main__":
    main()