import sys
from matplotlib import pyplot as plt, transforms
from matplotlib.patches import Ellipse, Circle
from scipy.stats import chi2

import numpy as np
from kf import KF, Gaussian, KFParams
from ekf import EKF, EKFParams

def parse_args():
    # sys.argv[0] is the script name itself
    if len(sys.argv) == 2:
        first_arg = sys.argv[1]
        return first_arg
    
    raise ValueError("Invalid arguments. Please provide a single argument corresponding to the task to run, e.g. 1c.")

def kf_config_v1():
    A = np.array([
        [1, 1],
        [0, 1]
    ], dtype=np.float64)
    R = np.array([
        [1/4, 1/2],
        [1/2, 1]
    ], dtype=np.float64)
    return KFParams(A, None, R, None, None)

def kf_config_v2():
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

def ekf_config_v1():
    def g(x, u):
        return np.array([
            x[0] + np.cos(x[2]),
            x[1] + np.sin(x[2]),
            x[2]
        ], dtype=np.float64)
    def G(x, u):
        return np.array([
            [1, 0, -np.sin(x[2])],
            [0, 1, np.cos(x[2])],
            [0, 0, 1]
        ], dtype=np.float64)
    return EKFParams(None, None, g, None, G, None)

def ekf_config_v2():
    def g(x, u):
        return np.array([
            x[0] + np.cos(x[2]),
            x[1] + np.sin(x[2]),
            x[2]
        ], dtype=np.float64)
    def G(x, u):
        return np.array([
            [1, 0, -np.sin(x[2])],
            [0, 1, np.cos(x[2])],
            [0, 0, 1]
        ], dtype=np.float64)
    def h(x):
        return x[0]
    def H(x):
        return np.array([
            [1, 0, 0]
        ], dtype=np.float64)
    Q = np.array([
        [0.01]
    ], dtype=np.float64)
    return EKFParams(None, Q, g, h, G, H)

def init_state_ex_1_and_2():
    return Gaussian(
        mu=np.array([0, 0], dtype=np.float64),
        Sigma=np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.float64)
    )

def init_state_ex_4ad():
    return Gaussian(
        mu=np.array([0, 0, 0], dtype=np.float64),
        Sigma=np.array([
            [0.01, 0, 0],
            [0, 0.01, 0],
            [0, 0, 10000]
        ], dtype=np.float64)
    )

def init_state_ex_4e():
    return Gaussian(
        mu=np.array([0, 0, 0], dtype=np.float64),
        Sigma=np.array([
            [0.01, 0, 0],
            [0, 10000, 0],
            [0, 0, 0.01]
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
    x = init_state_ex_1_and_2()
    filter = KF(kf_config_v1())

    ITERS = 5

    print(x)
    for _ in range(ITERS):
        x, _ = filter.update(x, None, None)
        print(x)

def f1d():
    x = init_state_ex_1_and_2()
    filter = KF(kf_config_v1())

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
    plt.savefig('3.8/out/1d.png')

def f1e():
    x = init_state_ex_1_and_2()
    filter = KF(kf_config_v1())

    ITERS = 10

    pccs = [float("nan")]

    for _ in range(ITERS):
        x, _ = filter.update(x, None, None)
        pccs.append(x.Sigma[0, 1] / np.sqrt(x.Sigma[0, 0] * x.Sigma[1, 1]))

    plt.plot(pccs)
    plt.xlabel('timestep')
    plt.ylabel('Pearson correlation coefficient')
    plt.savefig('3.8/out/1e.png')

def f2b():
    x = init_state_ex_1_and_2()
    filter = KF(kf_config_v2())

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
    plt.savefig('3.8/out/2b.png')

    print(beliefs[-1][1])
    print(beliefs[-1][0])

def f4a():
    x = init_state_ex_4ad()
    filter = EKF(ekf_config_v1())

    _, posterior = filter.update(x, None, None)

    _, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Draw uncertainty ellipse for initial state
    ax.scatter(x.mu[0], x.mu[1], c='b', marker='x')
    draw_uncertainty_ellipse(x.mu[:2], x.Sigma[:2,:2], ax, p=0.68, edgecolor='b', facecolor='b', alpha=0.1)

    # Draw uncertainty ellipse for posterior state
    ax.scatter(posterior.mu[0], posterior.mu[1], c='r', marker='x')
    draw_uncertainty_ellipse(posterior.mu[:2], posterior.Sigma[:2,:2], ax, p=0.68, edgecolor='r', facecolor='r', alpha=0.1)

    # Draw expected posterior distribution as circle
    circle = Circle((0, 0), radius=1, edgecolor='y', facecolor='none')
    ax.add_patch(circle)
    ax.set_aspect('equal')

    plt.xlabel('Position (x)')
    plt.ylabel('Position (y)')
    plt.savefig('3.8/out/4a.png')

    print("Initial state:")
    print(x)
    print("Posterior state:")
    print(posterior)

def f4d():
    x = init_state_ex_4ad()
    filter = EKF(ekf_config_v2())

    measurement = np.array([-0.7], dtype=np.float64)
    corr, pred = filter.update(x, None, measurement)

    _, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Draw measurement
    ax.axvline(x=measurement[0], color='g', alpha=0.5)

    # Draw intuitive state
    y_coord = np.sqrt(1 - measurement[0]**2)
    ax.scatter(measurement, y_coord, c='y', marker='x')
    ax.scatter(measurement, -y_coord, c='y', marker='x')
    circle = Circle((0, 0), radius=1, edgecolor='y', facecolor='none', alpha=0.5)
    ax.add_patch(circle)
    ax.set_aspect('equal')

    # Draw uncertainty ellipse for initial state
    ax.scatter(x.mu[0], x.mu[1], c='b', marker='x')
    draw_uncertainty_ellipse(x.mu[:2], x.Sigma[:2,:2], ax, p=0.68, edgecolor='b', facecolor='b', alpha=0.1)

    # Draw uncertainty ellipse for posterior state
    ax.scatter(pred.mu[0], pred.mu[1], c='b', marker='x')
    draw_uncertainty_ellipse(pred.mu[:2], pred.Sigma[:2,:2], ax, p=0.68, edgecolor='b', facecolor='b', alpha=0.1)

    # Draw uncertainty ellipse for state after incorporating measurement
    ax.scatter(corr.mu[0], corr.mu[1], c='b', marker='x')
    draw_uncertainty_ellipse(corr.mu[:2], corr.Sigma[:2,:2], ax, p=0.68, edgecolor='b', facecolor='b', alpha=0.1)

    plt.xlabel('Position (x)')
    plt.ylabel('Position (y)')
    plt.savefig('3.8/out/4d.png')

    print("Initial state:")
    print(x)
    print("Predicted state:")
    print(pred)
    print("Corrected state:")
    print(corr)

def f6():
    m = np.array([
        [1, 0.9, 0.99],
        [0.9, 1, 0.8],
        [0.99, 0.8, 1]
    ], dtype=np.float64)

    print(np.linalg.inv(m))
    

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
    elif task == "4a":
        f4a()
    elif task == "4d":
        f4d()
    elif task == "6":
        f6()

if __name__ == "__main__":
    main()