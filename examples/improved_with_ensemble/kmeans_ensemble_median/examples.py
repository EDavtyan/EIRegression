import sys
import os

from movies import run_multiple_executions as movies_example
from housing import run_multiple_executions as housing_example
from concrete import run_multiple_executions as concrete_example


def main():
    N_BUCKETS = 10
    BUCKETING = "quantile"
    ITTERATIONS = 50
    save_dir = "/home/edgar.davtyan/projects/recla_v1/examples/kmeans_median/results"

    # Run all three examples one after the other
    print("movies")
    movies_example(num_buckets=N_BUCKETS,
                   num_iterations=ITTERATIONS,
                   save_dir=os.path.join(save_dir, "movies"))
    print("housing")
    housing_example(num_buckets=N_BUCKETS,
                    num_iterations=ITTERATIONS,
                    save_dir=os.path.join(save_dir, "housing"))
    print("concrete")
    concrete_example(num_buckets=N_BUCKETS,
                     num_iterations=ITTERATIONS,
                     save_dir=os.path.join(save_dir, "concrete_2"))

if __name__ == '__main__':
    main()
