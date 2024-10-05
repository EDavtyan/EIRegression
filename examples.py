import os

from examples.XGBRegression.housing import run_multiple_executions as housing_example
# from examples.insurance import run_multiple_executions as insurance_example


def main( ):
    N_BUCKETS = 15
    BUCKETING = "quantile"
    ITTERATIONS = 50
    save_dir = "/home/edgar.davtyan/projects/recla_v1/slurm_jobs/results/"
    # save_dir = "/Users/eddavtyan/Documents/XAI/Projects/EIRegression/results"
    # print("Must be called as: python examples.py --<example_name> <n_buckets> <bucketing_method>")

    # Run all three examples one after the other
    # print("movies")
    # movies_example(num_buckets=N_BUCKETS,
    #                num_iterations=ITTERATIONS,
    #                save_dir=os.path.join(save_dir, "movies"))
    print("housing")
    housing_example(num_buckets=N_BUCKETS,
                    num_iterations=ITTERATIONS,
                    save_dir=os.path.join(save_dir, "housing"))
    # print("concrete")
    # concrete_example(num_buckets=N_BUCKETS,
    #                  num_iterations=ITTERATIONS,
    #                  save_dir=os.path.join(save_dir, "concrete"))

if __name__ == '__main__':
    main()
