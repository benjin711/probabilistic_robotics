from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Generator, Tuple

import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import concurrent.futures

from dataclasses import dataclass

@dataclass
class MinMaxNum:
    min: float
    max: float
    num: int


class SimpleGridPartitionsIterator:
    def __init__(self, partitions: "GridPartitions") -> None:
        self.partitions = partitions
        self.idxs = np.ndindex(partitions._ps.shape)

    def __iter__(self) -> "SimpleGridPartitionsIterator":
        return self

    def __next__(self) -> Tuple[Tuple[int], Tuple[Any], float]:
        idx = next(self.idxs)
        state = self.partitions.idx_to_state(idx)
        prob = self.partitions._ps[idx]

        return idx, state, prob
        
class RelevantPartitionsIterator:
    def __init__(self, partitions: "GridPartitions") -> None:
        self.partitions = partitions
        self.it = 0

    def __iter__(self) -> "RelevantPartitionsIterator":
        return self

    def __next__(self) -> Tuple[Tuple[int], Tuple[Any], float]:
        if self.it >= len(self.partitions.relevant_indices):
            raise StopIteration

        idx = self.partitions.relevant_indices[self.it]
        state = self.partitions.idx_to_state(idx)
        prob = self.partitions._ps[idx]
        self.it += 1

        return idx, state, prob
    
class GridPartitions:
    PROBABILITY_THRESHOLD = 0.9

    def __init__(
        self, 
        ivals: Tuple[MinMaxNum],
    ) -> None:
        self._ps = np.zeros([ival.num for ival in ivals])
        self._ivals = ivals
        self._deltas = [(ival.max - ival.min) / ival.num for ival in ivals]

        self.relevant_indices = []

    @property
    def ps(self) -> np.ndarray:
        return self._ps

    @property
    def ivals(self) -> Tuple[MinMaxNum]:
        return self._ivals

    def idx_to_state(self, idxs: Tuple[int]) -> Generator[float, None, None]:
        return (
            ival.min + idx * delta + delta / 2
            for ival, delta, idx in zip(self._ivals, self._deltas, idxs)
        )
    
    def state_to_idx(self, state: Generator[float, None, None]) -> Tuple[int]:
        state = list(state)
        
        # Return None if state is out of bounds
        if any(not (ival.min <= x <= ival.max) for ival, x in zip(self._ivals, state)):
            return None
        
        return tuple(
            int((x - ival.min) / delta)
            for ival, delta, x in zip(self._ivals, self._deltas, state)
        )

    def init_with_prior(self, prior: Callable) -> None:
        normalizer = 0

        for idx, state, _ in self.get_simple_partitions_iter():
            self._ps[idx] = prior(state)
            normalizer += self._ps[idx]

        for idx in np.ndindex(self._ps.shape):
            self._ps[idx] /= normalizer
    
    def get_simple_partitions_iter(self) -> SimpleGridPartitionsIterator:
        return SimpleGridPartitionsIterator(self)

    def _calc_relevant_indices(self) -> None:

        def _calc_relevance_threshold() -> None:
            # Flatten self.ps and sort it in descending order
            sorted_ps = np.sort(self._ps.flatten())[::-1]

            # Find the index when the cumulative sum exceeds the threshold
            cumsum = np.cumsum(sorted_ps)
            idx = np.where(cumsum > GridPartitions.PROBABILITY_THRESHOLD)[0][0]

            return sorted_ps[idx]

        self.relevant_indices = [tuple(idx) for idx in np.argwhere(
            self._ps >= _calc_relevance_threshold())]
    
    def get_relevant_partitions_iter(self) -> RelevantPartitionsIterator:
        if self.relevant_indices:
            return RelevantPartitionsIterator(self)

        self._calc_relevant_indices()
        return RelevantPartitionsIterator(self)
    
    def __getitem__(self, idx: Tuple[int]) -> float:
        return self._ps[idx]
    
    def __setitem__(self, idx: Tuple[int], value: float) -> None:
        self._ps[idx] = value
    
    def visualize_2D_histogram(
        self, 
        path: Path,
        slices: Tuple[Any] = (Ellipsis,),
        labels: Tuple[str] = ("X", "V", "p")
    ) -> None:
        # Create 2D array
        hist = self._ps[slices]
        assert len(hist.shape) == 2, "Histogram must be 2D"

        # Fetch the information from the relevant x and y axes
        x_idx, y_idx = 0, 1
        alt_idxs = [idx for idx, obj in enumerate(slices) if isinstance(obj, (slice, type(Ellipsis)))][:2]
        if len(alt_idxs) == 2:
            x_idx, y_idx = alt_idxs

        x_edges = np.linspace(self._ivals[x_idx].min, self._ivals[x_idx].max, self._ivals[x_idx].num, endpoint=False)
        y_edges = np.linspace(self._ivals[y_idx].min, self._ivals[y_idx].max, self._ivals[y_idx].num, endpoint=False)
        x_delta = self._deltas[x_idx]
        y_delta = self._deltas[y_idx]

        # Construct arrays for the anchor positions of the bars.
        xpos, ypos = np.meshgrid(x_edges + 0.5 * x_delta, y_edges + 0.5 * y_delta, indexing="ij")

        # Create a colormap
        dz = hist.ravel()
        cmap = plt.cm.get_cmap('jet')  # 'jet' is the name of the colormap
        max_height = np.max(dz)   # get the maximum bar height
        min_height = np.min(dz)
        rgba = [cmap((k-min_height)/max_height) for k in dz] 

        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(121, projection='3d', title="Histogram 3D Visualization")

        ax1.bar3d(
            xpos.ravel(), 
            ypos.ravel(), 
            0, 
            x_delta, 
            y_delta, 
            dz, 
            color=rgba, 
            zsort='average'
        )
        # Add axis labels
        ax1.set_xlabel(labels[0])
        ax1.set_ylabel(labels[1])
        ax1.set_zlabel(labels[2])

        ax2 = fig.add_subplot(122, aspect="equal", title="Histogram 2D Visualization")
        pc2 = ax2.pcolormesh(xpos, ypos, hist, cmap="jet")
        ax2.set_xlabel(labels[0])
        ax2.set_ylabel(labels[1])

        fig.colorbar(pc2, ax=ax2, orientation="vertical")

        plt.subplots_adjust(wspace=0.5)
        plt.savefig(path)


class Prior_ex1:
    def __init__(
        self,
        value: float,
        ivals: Tuple[MinMaxNum],
    ) -> None:
        self.value = value
        self.deltas = [(ival.max - ival.min) / ival.num for ival in ivals]

    def __call__(self, state: Tuple[float]) -> float:
        if all(-delta <= x <= delta for x, delta in zip(state, self.deltas)): 
            return self.value
        
        return 0
    

def prediction_func_3x1(
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


def measurement_func_ex1(curr_state: Generator[float, None, None], measurement: float):
    curr_state = list(curr_state)
    var = 10
    gaussian = norm(loc=curr_state[0], scale=np.sqrt(var))
    return gaussian.pdf(measurement)


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

class DiscreteBayesFilter:
    def __init__(
        self, 
        prediction_func: Callable = None,
        measurement_func: Callable = None
    ) -> None:
        self.prediction_func = prediction_func
        self.measurement_func = measurement_func

    def prediction(
        self,
        partitions: GridPartitions,
        control: float
    ) -> GridPartitions:
        new_partitions = GridPartitions(partitions.ivals)

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