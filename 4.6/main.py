from pathlib import Path
import sys
import time

import numpy as np

from histogram_filter_utils import Prior_ex1, GridPartitions, MinMaxNum, Prior_ex2
from dbf import DeterministicStateTransitionPredictor, DiscreteBayesFilter, BasicPredictor, measurement_func_ex2, prediction_func_ex1, measurement_func_ex1, state_transition_func_ex2

def parse_args():
    # sys.argv[0] is the script name itself
    if len(sys.argv) == 2:
        first_arg = sys.argv[1]
        return first_arg
    
    raise ValueError("Invalid arguments. Please provide a single argument corresponding to the task to run, e.g. 1c.")

def f1():
    PATH = Path("4.6/out/ex1")
    PATH.mkdir(parents=True, exist_ok=True)
    TIMESTEPS = 5
    MIN_X, MAX_X, NUM_X = -15, 15, 60
    MIN_Y, MAX_Y, NUM_Y = -15, 15, 60

    x_ivals = MinMaxNum(MIN_X, MAX_X, NUM_X)
    y_ivals = MinMaxNum(MIN_Y, MAX_Y, NUM_Y)
    ivals = (x_ivals, y_ivals)
    
    prior = Prior_ex1(0.25, ivals)
    partitions = GridPartitions(ivals)
    partitions.init_with_prior(prior)

    controls = [None] * 5
    measurements = [None] * 5
    measurements[-1] = 5

    predictor = BasicPredictor(prediction_func_ex1)
    dbf = DiscreteBayesFilter(predictor, measurement_func_ex1)

    evolution = [partitions]
    partitions.visualize_2D_histogram(PATH / f"f1a_vis_t{0}.png")

    for t in range(TIMESTEPS):
        print(f"t = {t+1}...")
        partitions, pred_partitions = dbf.update(partitions, measurements[t], controls[t])

        if t == TIMESTEPS - 1:
            pred_partitions.visualize_2D_histogram(PATH / f"f1a_vis_t{t+1}_pred.png")

        partitions.visualize_2D_histogram(PATH / f"f1a_vis_t{t+1}.png")
        evolution.append(partitions)

def f2():
    PATH = Path("4.6/out/ex2")
    PATH.mkdir(parents=True, exist_ok=True)
    TIMESTEPS = 1
    MIN_X, MAX_X, NUM_X = -2, 2, 40
    MIN_Y, MAX_Y, NUM_Y = -2, 2, 40
    MIN_THETA, MAX_THETA, NUM_THETA = -np.pi, np.pi, 72
    x_ivals = MinMaxNum(MIN_X, MAX_X, NUM_X)
    y_ivals = MinMaxNum(MIN_Y, MAX_Y, NUM_Y)
    theta_ivals = MinMaxNum(MIN_THETA, MAX_THETA, NUM_THETA)
    ivals = (x_ivals, y_ivals, theta_ivals)

    prior = Prior_ex2()
    partitions = GridPartitions(ivals)
    partitions.init_with_prior(prior)

    predictor = DeterministicStateTransitionPredictor(state_transition_func_ex2)
    dbf = DiscreteBayesFilter(predictor, measurement_func_ex2)

    def preprocess_func(ps):
        hist = np.sum(ps, axis=2)
        return hist, 0, 1

    partitions.visualize_2D_histogram(
        path=PATH / f"f2a_vis_t{0}.png",
        labels=("X", "Y", "p"),
        preprocess_func=preprocess_func
    )

    controls = [None]
    measurements = [-0.7]

    for t in range(TIMESTEPS):
        print(f"t = {t+1}...")
        partitions, pred_partition = dbf.update(partitions, measurements[t], controls[t])

        pred_partition.visualize_2D_histogram(
            path=PATH / f"f2a_vis_t{t+1}_pred.png",
            labels=("X", "Y", "p"),
            preprocess_func=preprocess_func
        )

        partitions.visualize_2D_histogram(
            path=PATH / f"f2a_vis_t{t+1}.png",
            labels=("X", "Y", "p"),
            preprocess_func=preprocess_func
        )


def main():
    start_time = time.time()

    task = parse_args()
    if task == "1":
        f1()
    elif task == "2":
        f2()

    end_time = time.time()
    print(f"The function took {end_time - start_time} seconds to execute.")

if __name__ == "__main__":
    main()