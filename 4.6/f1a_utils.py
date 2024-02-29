from copy import deepcopy
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import concurrent.futures

from dataclasses import dataclass

@dataclass
class State:
    x: float
    v: float


class Partitions2DIterator:
        def __init__(self, partitions: "Partitions2D") -> None:
            self.partitions = partitions
            self.it_x = 0
            self.it_y = 0

        def __next__(self) -> Tuple[int, int, float, float, float]:
            if self.it_x >= self.partitions.xnum:
                raise StopIteration

            state = self.partitions.idx_to_state(self.it_x, self.it_y)
            prob = self.partitions.ps[self.it_x, self.it_y]
            curr_it_x, curr_it_y = self.it_x, self.it_y

            self.it_y += 1
            if self.it_y >= self.partitions.ynum:
                self.it_y = 0
                self.it_x += 1

            return curr_it_x, curr_it_y, state[0], state[1], prob
    
class Partitions2D:
    def __init__(
        self, 
        xmin: float, 
        xmax: float, 
        xnum: float, 
        ymin: float, 
        ymax: float, 
        ynum: float,
        prior: Callable
    ) -> None:
        self.ps = np.zeros((xnum, ynum))
        self.xmin = xmin
        self.xmax = xmax
        self.xnum = xnum
        self.ymin = ymin
        self.ymax = ymax
        self.ynum = ynum
        self.x_delta = (self.xmax - self.xmin) / self.xnum
        self.y_delta = (self.ymax - self.ymin) / self.ynum

        self.init_ps(prior) if prior else None

    def idx_to_state(self, it_x: int, it_y: int) -> Tuple[float, float]:
        return (
            self.xmin + it_x * self.x_delta + self.x_delta / 2, 
            self.ymin + it_y * self.y_delta + self.y_delta / 2
        )

    def state_to_idx(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(round((x - self.x_delta / 2 - self.xmin) / self.x_delta)), 
            int(round((y - self.y_delta / 2 - self.ymin) / self.y_delta))
        )

    def init_ps(self, prior: Callable) -> None:
        normalizer = 0
        for it_x in range(self.ps.shape[0]):
            for it_y in range(self.ps.shape[1]):
                state = self.idx_to_state(it_x, it_y)
                self.ps[it_x, it_y] = prior(*state)
                normalizer += self.ps[it_x, it_y]

        for it_x in range(self.ps.shape[0]):
            for it_y in range(self.ps.shape[1]):
                self.ps[it_x, it_y] /= normalizer
    
    def __iter__(self):
        return Partitions2DIterator(self)
    
    def __getitem__(self, index: Tuple[int, int]) -> float:
        return self.ps[index[0], index[1]]
    
    def __setitem__(self, index: Tuple[int, int], value: float) -> None:
        self.ps[index[0], index[1]] = value
    
    def visualize_2D_histogram(self, path: Path) -> None:
        # Create a 2D histogram
        hist = self.ps
        xedges = np.linspace(self.xmin, self.xmax, self.xnum, endpoint=False)
        yedges = np.linspace(self.ymin, self.ymax, self.ynum, endpoint=False)

        # Construct arrays for the anchor positions of the bars.
        xpos, ypos = np.meshgrid(xedges + self.x_delta, yedges + self.y_delta, indexing="ij")

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
            self.x_delta, 
            self.y_delta, 
            dz, 
            color=rgba, 
            zsort='average'
        )
        # Add axis labels
        ax1.set_xlabel('X')
        ax1.set_ylabel('V')
        ax1.set_zlabel('P')

        ax2 = fig.add_subplot(122, aspect="equal", title="Histogram 2D Visualization")
        pc2 = ax2.pcolormesh(xpos, ypos, hist, cmap="jet")
        ax2.set_xlabel('X')
        ax2.set_ylabel('V')

        fig.colorbar(pc2, ax=ax2, orientation="vertical")

        plt.subplots_adjust(wspace=0.5)
        plt.savefig(path)

    

class Prior:
    def __init__(
        self, 
        xmin: float, 
        xmax: float, 
        xnum: float, 
        ymin: float, 
        ymax: float, 
        ynum: float,
    ) -> None:
        self.x_delta = (xmax - xmin) / xnum
        self.y_delta = (ymax - ymin) / ynum

    def __call__(self, x: float, v: float) -> float:
        if -self.x_delta <= x <= self.x_delta and -self.y_delta <= v <= self.y_delta:
            return 0.25
        
        return 0
    

def prediction_func(curr_state: State, prev_state: State, control: float) -> float:
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
    mean = A @ np.array([[prev_state.x], [prev_state.v]])
    mean = mean.T.squeeze().tolist()
    mvg = multivariate_normal(mean, cov, allow_singular=True)

    return mvg.pdf([curr_state.x, curr_state.v])


def measurement_func(curr_state: State, measurement: float):
    var = 10
    gaussian = norm(loc=curr_state.x, scale=np.sqrt(var))
    return gaussian.pdf(measurement)


def _calc_partition_probability(args):
    it_x, it_v, curr_x, curr_v, partitions, control, prediction_func = args
    partition_probability = 0.0

    for _, _, prev_x, prev_v, prev_p in partitions:
        if prev_p == 0:
            continue

        partition_probability += prediction_func(
            State(curr_x, curr_v), 
            State(prev_x, prev_v),
            control
        ) * prev_p

    return it_x, it_v, partition_probability

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
        partitions: Partitions2D,
        control: float
    ) -> Partitions2D:
        new_partitions = deepcopy(partitions)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            params = [
                (it_x, it_v, curr_x, curr_v, partitions, control, self.prediction_func)
                for it_x, it_v, curr_x, curr_v, _ in new_partitions
            ]
            results = executor.map(_calc_partition_probability, params)

            for it_x, it_v, p in results:
                new_partitions[it_x, it_v] = p
        
        return new_partitions
    
    def correction(
        self,
        partitions: Partitions2D,
        measurement: float,
    ) -> Partitions2D:
        if measurement is None:
            return partitions

        normalizer = 0
        for it_x, it_v, curr_x, curr_v, _ in partitions:
            partitions[it_x, it_v] *= self.measurement_func(
                State(curr_x, curr_v), 
                measurement
            )

            normalizer += partitions[it_x, it_v]

        for it_x, it_v, _, _, _ in partitions:
            partitions[it_x, it_v] /= normalizer
        
        return partitions


    def update(
        self,
        partitions: Partitions2D, 
        measurement: float = None, 
        control: float = None
    ) -> Partitions2D:
        
        partitions = self.prediction(partitions, control)
        pred_partitions = deepcopy(partitions)
        
        partitions = self.correction(partitions, measurement)

        return partitions, pred_partitions