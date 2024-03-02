from pathlib import Path
import sys
import time

from f1a_utils import Prior, Partitions2D, DiscreteBayesFilter, MinMaxNum, measurement_func, prediction_func

def parse_args():
    # sys.argv[0] is the script name itself
    if len(sys.argv) == 2:
        first_arg = sys.argv[1]
        return first_arg
    
    raise ValueError("Invalid arguments. Please provide a single argument corresponding to the task to run, e.g. 1c.")

def f1():
    TIMESTEPS = 5
    MIN_X, MAX_X, NUM_X = -15, 15, 30
    MIN_Y, MAX_Y, NUM_Y = -15, 15, 30
    x_ivals = MinMaxNum(MIN_X, MAX_X, NUM_X)
    y_ivals = MinMaxNum(MIN_Y, MAX_Y, NUM_Y)
    
    prior = Prior(x_ivals, y_ivals)
    partitions = Partitions2D(x_ivals, y_ivals)
    partitions.init_with_prior(prior)

    controls = [None] * 5
    measurements = [None] * 5
    measurements[-1] = 5

    dbf = DiscreteBayesFilter(prediction_func, measurement_func)

    evolution = [partitions]
    partitions.visualize_2D_histogram(Path(f"4.6/out/f1a_vis_t{0}.png"))

    for t in range(TIMESTEPS):
        partitions, pred_partitions = dbf.update(partitions, measurements[t], controls[t])

        if t == TIMESTEPS - 1:
            pred_partitions.visualize_2D_histogram(Path(f"4.6/out/f1a_vis_t{t+1}_pred.png"))

        partitions.visualize_2D_histogram(Path(f"4.6/out/f1a_vis_t{t+1}.png"))
        evolution.append(partitions)


def main():
    start_time = time.time()

    task = parse_args()
    if task == "1" or task == "1a" or task == "1b":
        f1()

    end_time = time.time()
    print(f"The function took {end_time - start_time} seconds to execute.")

if __name__ == "__main__":
    main()