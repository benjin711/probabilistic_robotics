from pathlib import Path
from typing import Any, Callable, Generator, Tuple

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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
        labels: Tuple[str] = ("X", "V", "p"),
        preprocess_func: Callable = None
    ) -> None:
        # Create 2D array
        hist, x_idx, y_idx = preprocess_func(self._ps) if preprocess_func else self._ps
        assert len(hist.shape) == 2, "Histogram must be 2D"

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
        plt.close()


class Prior_ex1:
    def __init__(
        self,
        value: float,
        ivals: Tuple[MinMaxNum],
    ) -> None:
        self.value = value
        self.deltas = [(ival.max - ival.min) / ival.num for ival in ivals]

    def __call__(self, state: Generator[float, None, None]) -> float:
        if all(-delta <= x <= delta for x, delta in zip(state, self.deltas)): 
            return self.value
        
        return 0
    

class Prior_ex2:
    def __init__(self) -> None:
        mean = np.array([0, 0, 0])
        cov = np.array([
            [0.01, 0, 0],
            [0, 0.01, 0],
            [0, 0, 10000]
        ])
        self.mvg = multivariate_normal(mean, cov)

    def __call__(self, state: Generator[float, None, None]) -> float:
        state = list(state)
        return self.mvg.pdf(state)
