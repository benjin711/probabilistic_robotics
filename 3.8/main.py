import sys

import numpy as np
from kf import KF, Gaussian, KFParams

def parse_args():
    # sys.argv[0] is the script name itself
    if len(sys.argv) == 2:
        first_arg = sys.argv[1]
        return first_arg
    
    raise ValueError("Invalid arguments. Please provide a single argument corresponding to the task to run, e.g. 1c.")

def f1c():
    x = Gaussian(
        mu=np.array([0, 0], dtype=np.float64),
        Sigma=np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.float64)
    )
    A = np.array([
        [1, 1],
        [0, 1]
    ], dtype=np.float64)
    R = np.array([
        [1/4, 0],
        [0, 1]
    ], dtype=np.float64)

    filter = KF(KFParams(A, None, R, None, None))

    ITERS = 5

    print(x)
    for _ in range(ITERS):
        x, _ = filter.update(x, None, None)
        print(x)

def main():
    task = parse_args()
    if task == "1c":
        f1c()

if __name__ == "__main__":
    main()