from pathlib import Path
import sys
import time

from histogram_filter_utils import Prior_ex1, GridPartitions, DiscreteBayesFilter, MinMaxNum, measurement_func_ex1, prediction_func_3x1

def parse_args():
    # sys.argv[0] is the script name itself
    if len(sys.argv) == 2:
        first_arg = sys.argv[1]
        return first_arg
    
    raise ValueError("Invalid arguments. Please provide a single argument corresponding to the task to run, e.g. 1c.")

def f1():
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

    dbf = DiscreteBayesFilter(prediction_func_3x1, measurement_func_ex1)

    evolution = [partitions]
    partitions.visualize_2D_histogram(Path(f"4.6/out/f1a_vis_t{0}.png"))

    for t in range(TIMESTEPS):
        print(f"t = {t}...")
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