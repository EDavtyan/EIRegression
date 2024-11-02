import os, sys
import numpy as np
import pandas as pd
import json

sys.path.append('/home/davtyan.edd/projects/EIRegression/')
os.chdir('/home/davtyan.edd/projects/EIRegression/')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix

from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter
from EIRegressor.utils import compute_weighted_accuracy
from rule_analysis.compare_rules import process_rule_similarities
from examples.rule_analysis_experiments.sim_def_V2.utils import compute_and_save_average_similarity_matrix, run_experiments_for_buckets, update_results_with_weighted_accuracy




def run_multiple_executions(save_dir, num_buckets, num_iterations, dataset_name):
    os.makedirs(save_dir, exist_ok=True)
    all_results_file_path = os.path.join(save_dir,
                                         f"{dataset_name}_results_{num_buckets}_buckets_{num_iterations}_iterations.json")

    all_results = {}
    # Check if the consolidated results file exists and load it
    if os.path.exists(all_results_file_path):
        with open(all_results_file_path, 'r') as json_file:
            all_results = json.load(json_file)

    for n_buckets in range(2, num_buckets + 1):
        bucket_results = all_results.get(f"{n_buckets}_buckets", [])

        # Run experiments for the current number of buckets
        bucket_results = run_experiments_for_buckets(save_dir, n_buckets, num_iterations, bucket_results)

        # Process rule similarities after all iterations for the current number of buckets
        # process_rule_similarities(rules_dir=os.path.join(save_dir, "rules"),
                                #   dataset_name=f"{dataset_name}",
                                #   out_base=os.path.join(save_dir, "rule_similarities"))

        # Compute weighted accuracy and update results
        # update_results_with_weighted_accuracy(save_dir, bucket_results, n_buckets)

        # Compute average similarity matrix over all iterations
        compute_and_save_average_similarity_matrix(save_dir, bucket_results, n_buckets)

        # Update the results file
        all_results[f"{n_buckets}_buckets"] = bucket_results
        with open(all_results_file_path, 'w') as json_file:
            json.dump(all_results, json_file, indent=4)


if __name__ == '__main__':
    dataset_name = "concrete_3_breaks_V2_3_diagonal_ones"

    save_dir = f"examples/rule_analysis_experiments/sim_def_V2/results/{dataset_name}"
    run_multiple_executions(save_dir=save_dir,
                            num_buckets=10,
                            num_iterations=25,
                            dataset_name=dataset_name)