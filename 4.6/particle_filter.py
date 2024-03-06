from copy import deepcopy
from typing import Callable, Protocol
import numpy as np
from scipy.stats import norm

from particle_filter_utils import ParticleSet

class Resampler(Protocol):
    
    def __call__(
        self, 
        rng: np.random.Generator, 
        particle_set: ParticleSet
    ) -> ParticleSet:
        ...

class RandomResampler:

    def __call__(
        self, 
        rng: np.random.Generator, 
        particle_set: ParticleSet
    ) -> ParticleSet:

        weights = particle_set.weights
        weights /= np.sum(weights)
        indices = rng.choice(
            np.arange(len(particle_set)), 
            len(particle_set), 
            p=weights
        )
        particles, weights = particle_set[indices]
        particle_set.particles = particles
        particle_set.weights = weights

        # Reset weights
        particle_set.weights = np.ones_like(particle_set.weights)

        return particle_set

class LowVarianceResampler:

    def __call__(
        self, 
        rng: np.random.Generator, 
        particle_set: ParticleSet
    ) -> ParticleSet:

        raise NotImplementedError


def state_transition_func_ex4(
    rng: np.random.Generator, 
    state: np.ndarray, 
    control: float
) -> np.ndarray:
    A = np.array([
        [1, 1], 
        [0, 1]
    ])
    state = A @ state

    epsilon = rng.normal(0, 1) 
    state[0] += 0.5 * epsilon
    state[1] += epsilon

    return state
    

def measurement_func_ex4(
    state: np.ndarray,
    measurement: float
) -> float:
    var = 10
    gaussian = norm(loc=state[0], scale=np.sqrt(var))
    return gaussian.pdf(measurement)


def state_transition_func_ex5(
    rng: np.random.Generator, 
    state: np.ndarray, 
    control: float
) -> np.ndarray:
    return np.array([
        state[0] + np.cos(state[2]),
        state[1] + np.sin(state[2]),
        state[2]
    ])

def measurement_func_ex5(
    state: np.ndarray,
    measurement: float
) -> float:
    var = 0.01
    gaussian = norm(loc=state[0], scale=np.sqrt(var))
    return gaussian.pdf(measurement)
    

class ParticleFilter:

    def __init__(
        self,
        rng: np.random.Generator,
        state_transition_func: Callable,
        measurement_func: Callable = None,
        resampler: Resampler = None,
    ) -> None:
        self.rng = rng
        self.state_transition_func = state_transition_func
        self.measurement_func = measurement_func
        self.resampler = resampler

    def propagate(self, particle_set: ParticleSet, control: float) -> ParticleSet:

        for idx, (particle, weight) in enumerate(particle_set):
            particle = self.state_transition_func(self.rng, particle, control)
            particle_set[idx] = particle, weight

        return particle_set
    
    def update_weights(self, particle_set: ParticleSet, measurement: float) -> ParticleSet:
        if measurement is None:
            return particle_set

        for idx, (particle, weight) in enumerate(particle_set):
            weight *= self.measurement_func(particle, measurement)
            particle_set[idx] = particle, weight

        return particle_set

    def update(
        self, 
        particle_set: ParticleSet, 
        measurement: float = None, 
        control: float = None
    ) -> ParticleSet:
        
        particle_set = self.propagate(particle_set, control)

        if self.measurement_func is not None:
            particle_set = self.update_weights(particle_set, measurement)

        particle_set_pred = deepcopy(particle_set)
        
        if self.resampler is not None:
            particle_set = self.resampler(self.rng, particle_set)

        return particle_set, particle_set_pred
