import sys

from examples.movies import execute as movies_example
from examples.housing import execute as housing_example
from examples.insurance import execute as insurance_example
from examples.concrete import execute as concrete_example


def main():
    N_BUCKETS = 3
    BUCKETING = "quantile"
    if len(sys.argv) < 2:
        print("Must be called as: python examples.py --<example_name> <n_buckets> <bucketing_method>")
    if len(sys.argv) > 2:
        N_BUCKETS = int(sys.argv[2])
    if len(sys.argv) > 3:
        BUCKETING = sys.argv[3]
    if sys.argv[1] == "--movies":
        movies_example(N_BUCKETS, BUCKETING)
    if sys.argv[1] == "--housing":
        housing_example(N_BUCKETS, BUCKETING)
    if sys.argv[1] == "--insurance":
        insurance_example(N_BUCKETS, BUCKETING)
    if sys.argv[1] == "--concrete":
        concrete_example(N_BUCKETS, BUCKETING)


if __name__ == '__main__':
    main()
