

from typing import Callable, Tuple
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


class ParticleSet:

    def __init__(
        self,
        num_particles: int,
        state_dim: int
    ) -> None:
        self.particles = np.zeros((num_particles, state_dim))
        self.weights = np.ones((num_particles))

    def gaussian_init(
        self,
        rng: np.random.Generator,
        mean: np.ndarray,
        cov: np.ndarray
    ) -> None:
        self.particles = rng.multivariate_normal(mean, cov, self.particles.shape[0])

    def visualize_particles(
        self,
        path: Path,
        labels: Tuple[str] = ("X", "V"),
        preprocess_func: Callable = None
    ) -> None:
        particles, weights = preprocess_func(self.particles, self.weights) \
            if preprocess_func else (self.particles, self.weights)
        
        fig = plt.figure(figsize=(8, 6))

        num_particles, nd = particles.shape
        particles = particles.T.tolist()

        ax1 = fig.add_subplot(
            111, 
            aspect='equal' if nd == 2 else 'auto',
            projection='3d' if nd == 3 else None,
            title=f"{nd}D Scatter Visualization M={num_particles}"
        )

        sc1 = ax1.scatter(*particles, c=weights, cmap='jet')
        ax1.set_xlabel(labels[0])
        ax1.set_ylabel(labels[1])
        if nd == 3:
            ax1.set_zlabel(labels[2])

        cbar = fig.colorbar(sc1, label='Weight', orientation='vertical')

        # Add some horizontal space between the colorbar and the plot
        fig.subplots_adjust(right=0.9)
        cbar.ax.set_position([0.85, .15, .05, .7])

        plt.savefig(path)
        plt.close()

        


