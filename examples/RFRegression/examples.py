import sys
import os

sys.path.append('/home/edgar.davtyan/projects/recla_v1')

from movies import run_multiple_executions as movies_example
from housing import run_multiple_executions as housing_example
from concrete import run_multiple_executions as concrete_example
from insurance import run_multiple_executions as insurance_example
from house_16 import run_multiple_executions as house_16_example
from bank32NH import run_multiple_executions as bank_32_example


def main():
    BUCKETING = "quantile"

    save_dir = "/home/edgar.davtyan/projects/recla_v1/examples/RFRegression/results/"
    # save_dir = "/Users/eddavtyan/Documents/XAI/Projects/EIRegression/examples/MLP_regression/results/"
    # print("Must be called as: python examples.py --<example_name> <n_buckets> <bucketing_method>")

    # Run all three examples one after the other
    # print("movies")
    # movies_example(num_buckets=10,
    #                num_iterations=3,
    #                save_dir=os.path.join(save_dir, "movies_5_breaks"),
    #                single_rules_breaks=5)
    # print("housing")
    # housing_example(num_buckets=10,
    #                 num_iterations=3,
    #                 save_dir=os.path.join(save_dir, "housing_5_breaks"),
    #                 single_rules_breaks=5)
    # print("concrete")
    # concrete_example(num_buckets=10,
    #                  num_iterations=3,
    #                  save_dir=os.path.join(save_dir, "concrete_5_breaks"),
    #                  single_rules_breaks=5)
    # print("insurance")
    # insurance_example(num_buckets=10,
    #                   num_iterations=3,
    #                   save_dir=os.path.join(save_dir, "insurance_5_breaks"),
    #                   single_rules_breaks=5)
    print("house_16H_5_breaks")
    house_16_example(num_buckets=10,
                     num_iterations=3,
                     save_dir=os.path.join(save_dir, "house_16H_5_breaks"),
                     single_rules_breaks=5)
    print("house_16H_3_breaks")
    house_16_example(num_buckets=10,
                     num_iterations=3,
                     save_dir=os.path.join(save_dir, "house_16H_5_breaks"),
                     single_rules_breaks=3)
    # print("bank32NH_3_breaks")
    # bank_32_example(num_buckets=15,
    #                 num_iterations=1,
    #                 save_dir=os.path.join(save_dir, "bank32NH_3_breaks"),
    #                 single_rules_breaks=3)
    # print("bank32NH_5_breaks")
    # bank_32_example(num_buckets=15,
    #                 num_iterations=3,
    #                 save_dir=os.path.join(save_dir, "bank32NH_5_breaks"),
    #                 single_rules_breaks=5)

if __name__ == '__main__':
    main()