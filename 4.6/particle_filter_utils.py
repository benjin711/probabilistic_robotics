

from typing import Callable, Tuple
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


class SimpleParticleSetIterator:
    def __init__(self, particle_set: "ParticleSet") -> None:
        self.particle_set = particle_set
        self.it = 0

    def __iter__(self) -> "SimpleParticleSetIterator":
        return self

    def __next__(self) -> Tuple[np.ndarray, float]:
        if self.it >= len(self.particle_set.particles):
            raise StopIteration

        particle = self.particle_set.particles[self.it]
        weight = self.particle_set.weights[self.it]
        self.it += 1

        return particle, weight

class ParticleSet:

    def __init__(
        self,
        num_particles: int,
        state_dim: int
    ) -> None:
        self._particles = np.zeros((num_particles, state_dim))
        self._weights = np.ones((num_particles))

    def __len__(self) -> int:
        return len(self._particles)

    @property
    def particles(self) -> np.ndarray:
        return self._particles
    
    @particles.setter
    def particles(self, particles: np.ndarray) -> None:
        assert particles.shape == self._particles.shape, "Invalid shape"
        self._particles = particles
    
    @property
    def weights(self) -> np.ndarray:
        return self._weights
    
    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        assert weights.shape == self._weights.shape, "Invalid shape"
        self._weights = weights

    def gaussian_init(
        self,
        rng: np.random.Generator,
        mean: np.ndarray,
        cov: np.ndarray
    ) -> None:
        self._particles = rng.multivariate_normal(mean, cov, self._particles.shape[0])

    def __iter__(self):
        return SimpleParticleSetIterator(self)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        return self._particles[idx], self._weights[idx]
    
    def __setitem__(self, idx: int, value: Tuple[np.ndarray, float]) -> None:
        self._particles[idx], self._weights[idx] = value

    def visualize_particles(
        self,
        path: Path,
        labels: Tuple[str] = ("X", "V"),
        xlim: Tuple[float] = (-15, 15),
        ylim: Tuple[float] = (-15, 15),
        preprocess_func: Callable = None
    ) -> None:
        particles, weights = preprocess_func(self._particles, self._weights) \
            if preprocess_func else (self._particles, self._weights)
        
        fig = plt.figure(figsize=(8, 6))

        num_particles, nd = particles.shape
        particles = particles.T.tolist()

        ax1 = fig.add_subplot(
            111, 
            aspect='equal' if nd == 2 else 'auto',
            projection='3d' if nd == 3 else None,
            title=f"{nd}D Scatter Visualization M={num_particles}"
        )
        ax1.set_xlim(*xlim)
        ax1.set_ylim(*ylim)

        sc1 = ax1.scatter(*particles, c=weights, cmap='viridis', s=1)
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

        


