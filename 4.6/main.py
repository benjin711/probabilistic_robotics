from pathlib import Path
import sys

from f1a_utils import Prior, Partitions2D, DiscreteBayesFilter, measurement_func, prediction_func

def parse_args():
    # sys.argv[0] is the script name itself
    if len(sys.argv) == 2:
        first_arg = sys.argv[1]
        return first_arg
    
    raise ValueError("Invalid arguments. Please provide a single argument corresponding to the task to run, e.g. 1c.")

def f1a():
    TIMESTEPS = 5
    MIN_X, MAX_X, NUM_X = -10, 10, 20
    MIN_Y, MAX_Y, NUM_Y = -10, 10, 20
    prior = Prior(MIN_X, MAX_X, NUM_X, MIN_Y, MAX_Y, NUM_Y)
    partitions = Partitions2D(MIN_X, MAX_X, NUM_X, MIN_Y, MAX_Y, NUM_Y, prior)

    controls = [None] * 5
    measurements = [None] * 5

    dbf = DiscreteBayesFilter(prediction_func, measurement_func)

    evolution = [partitions]
    partitions.visualize_2D_histogram(Path(f"4.6/out/f1a_vist{0}.png"))

    for t in range(TIMESTEPS):
        partitions = dbf.update(partitions, measurements[t], controls[t])
        partitions.visualize_2D_histogram(Path(f"4.6/out/f1a_vist{t+1}.png"))
        evolution.append(partitions)


def main():
    task = parse_args()
    if task == "1a":
        f1a()
        

if __name__ == "__main__":
    main()