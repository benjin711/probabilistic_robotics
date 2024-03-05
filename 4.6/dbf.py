from copy import deepcopy
from typing import Callable, Generator, Protocol, Tuple

import numpy as np
from scipy.stats import multivariate_normal, norm
import concurrent.futures

from histogram_filter_utils import GridPartitions

class Predictor(Protocol):

    def __call__(
        self,
        new_partitions: GridPartitions,
        partitions: GridPartitions,
        control: float
    ) -> float:
        ...


def prediction_func_ex1(
    curr_state: Generator[float, None, None], 
    prev_state: Generator[float, None, None], 
    control: float
) -> float:
    curr_state = list(curr_state)
    prev_state = list(prev_state)

    A = np.array([
        [1, 1],
        [0, 1]
    ])

    # Having perfectly correlated RV can lead to issues, e.g. cases where the multivariate
    # gaussian will always yield 0 due discrete nature of the representative states
    # and the density lying on a line. Adding an epsilon makes the experiment more realistic 
    # and makes the belief look more like a Gaussian avoiding weird corner cases
    EPSILON = 0.05
    cov = [[1/4, 1/2 - EPSILON],[1/2 - EPSILON, 1]]
    mean = A @ np.array([[prev_state[0]], [prev_state[1]]])
    mean = mean.T.squeeze().tolist()
    mvg = multivariate_normal(mean, cov, allow_singular=True)

    return mvg.pdf([curr_state[0], curr_state[1]])


def _calc_partition_probability(args):
    idx, curr_state, partitions, control, prediction_func = args
    partition_probability = 0.0

    for _, prev_state, prev_p in partitions.get_relevant_partitions_iter():
        partition_probability += prediction_func(
            curr_state, 
            prev_state,
            control
        ) * prev_p

    return idx, partition_probability


class BasicPredictor:

    def __init__(self, prediction_func: Callable) -> None:
        self.prediction_func = prediction_func

    def __call__(
        self,
        new_partitions: GridPartitions,
        partitions: GridPartitions,
        control: float
    ) -> GridPartitions:

        normalizer = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            params = [
                (idx, list(state), partitions, control, self.prediction_func)
                for idx, state, _ in new_partitions.get_simple_partitions_iter()
            ]
            results = executor.map(_calc_partition_probability, params)

            for idx, p in results:
                new_partitions[idx] = p
                normalizer += p
        
        for idx, _, _ in new_partitions.get_simple_partitions_iter():
            new_partitions[idx] /= normalizer
        
        return new_partitions


def measurement_func_ex1(curr_state: Generator[float, None, None], measurement: float):
    curr_state = list(curr_state)
    var = 10
    gaussian = norm(loc=curr_state[0], scale=np.sqrt(var))
    return gaussian.pdf(measurement)


def state_transition_func_ex2(
    curr_state: Generator[float, None, None], 
    control: float
) -> Generator[float, None, None]:
    curr_state = list(curr_state)
    
    new_state = [
        curr_state[0] + np.cos(curr_state[2]),
        curr_state[1] + np.sin(curr_state[2]),
        curr_state[2]
    ]

    for ns in new_state:
        yield ns


class DeterministicStateTransitionPredictor:

    def __init__(self, state_transition_func: Callable) -> None:
        self.state_transition_func = state_transition_func

    def __call__(
        self,
        new_partitions: GridPartitions,
        partitions: GridPartitions,
        control: float
    ) -> GridPartitions:

        normalizer = 0
        for idx, curr_state, curr_p in partitions.get_relevant_partitions_iter():
            curr_state = list(curr_state)
            new_state = self.state_transition_func(curr_state, control)
            new_idx = new_partitions.state_to_idx(new_state)
            if new_idx is None:
                continue

            new_partitions[new_idx] += curr_p
            normalizer += curr_p
        
        for idx, _, _ in new_partitions.get_simple_partitions_iter():
            new_partitions[idx] /= normalizer
        
        return new_partitions


def measurement_func_ex2(curr_state: Generator[float, None, None], measurement: float):
    curr_state = list(curr_state)
    var = 0.01
    gaussian = norm(loc=curr_state[0], scale=np.sqrt(var))
    return gaussian.pdf(measurement)


class DiscreteBayesFilter:
    def __init__(
        self, 
        predictor: Predictor = None,
        measurement_func: Callable = None
    ) -> None:
        self.predictor = predictor
        self.measurement_func = measurement_func

    def prediction(
        self,
        partitions: GridPartitions,
        control: float
    ) -> GridPartitions:
        new_partitions = GridPartitions(partitions.ivals)

        return self.predictor(new_partitions, partitions, control)
    
    def correction(
        self,
        partitions: GridPartitions,
        measurement: float,
    ) -> GridPartitions:
        if measurement is None:
            return partitions

        normalizer = 0
        for idx, curr_state, _ in partitions.get_simple_partitions_iter():
            partitions[idx] *= self.measurement_func(
                curr_state, 
                measurement
            )

            normalizer += partitions[idx]

        for idx, _, _ in partitions.get_simple_partitions_iter():
            partitions[idx] /= normalizer
        
        return partitions


    def update(
        self,
        partitions: GridPartitions, 
        measurement: float = None, 
        control: float = None
    ) -> GridPartitions:
        
        partitions = self.prediction(partitions, control)
        pred_partitions = deepcopy(partitions)
        
        partitions = self.correction(partitions, measurement)

        return partitions, pred_partitions