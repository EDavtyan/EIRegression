import sys
import pandas as pd
import os
import json

from rule_analysis.compare_rules import process_rule_similarities
sys.path.append('/home/davtyan.edd/projects/EIRegression/')
os.chdir('/home/davtyan.edd/projects/EIRegression/')

from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter

# from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
from EIRegressor.model_optimizer import ModelOptimizer


def execute(save_dir, similarity_matrices_dir, n_buckets=3, i=None, bucketing_method="quantile", statistic="median"):
    # Load dataframe
    data = pd.read_csv("examples/datasets/concrete_data.csv")
    target = "concrete_compressive_strength"
    data = data[data[target].notna()]

    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)

    # Creation of EI Regression with XGBoost
    eiReg = EmbeddedInterpreter(n_buckets=n_buckets,
                                bucketing_method=bucketing_method,
                                statistic=statistic,
                                max_iter=4000, lossfn="MSE",
                                min_dloss=0.0001, lr=0.005, precompute_rules=True,
                                force_precompute=True, device="cpu")

    eiReg.fit(X_train, y_train,
              add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
              column_names=data.drop(target, axis=1).columns)
    
    # Get predictions and buckets
    buck_pred, y_pred = eiReg.predict(X_test, return_buckets=True)
    
    # Save model rules to process them for the weighted accuracy
    save_results = os.path.join(save_dir, "rules")
    os.makedirs(save_results, exist_ok=True)
    eiReg.rules_to_txt(os.path.join(save_results, f"rule_results_{n_buckets}_buckets_{i}_iterations.txt"))

    # Compute Weighted Accuracy
    weighted_accuracy = eiReg.compute_weighted_accuracy(
        actual_values=y_test,
        predicted_buckets=buck_pred,
        similarity_matrices_dir=similarity_matrices_dir
    )

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    acc, f1, cm = eiReg.evaluate_classifier(X_test, y_test)

    top_uncertainties = eiReg.get_top_uncertainties()

    results = {"R2": r2,
               "MAE": mae,
               "MSE": mse,
               "Accuracy": acc,
               "F1": f1,
               "Weighted Accuracy": weighted_accuracy,
               "Confusion Matrix": cm.tolist(),
               "Uncertainties": top_uncertainties}
    
    # Save rules again, buth with results this time
    eiReg.rules_to_txt(os.path.join(save_results, f"rule_results_{n_buckets}_buckets_{i}_iterations.txt"),
                       results=results)

    return results


def run_multiple_executions(save_dir, num_buckets, num_iterations, similarity_matrices_dir, dataset_name):
    os.makedirs(save_dir, exist_ok=True)
    all_results_file_path = os.path.join(save_dir,
                                         f"results_{num_buckets}_buckets_{num_iterations}_iterations.json")

    all_results = {}
    # Check if the consolidated results file exists and load it
    if os.path.exists(all_results_file_path):
        with open(all_results_file_path, 'r') as json_file:
            all_results = json.load(json_file)

    for n_buckets in range(2, num_buckets + 1):
        bucket_results = all_results.get(f"{n_buckets}_buckets", [])

        for iteration in range(1, num_iterations + 1):
            # Construct the expected path for the results of this iteration
            expected_result_path = os.path.join(save_dir, "rules", f"rule_results_{n_buckets}_buckets_{iteration}_iterations.txt")

            # Check if this experiment's results already exist
            if not os.path.exists(expected_result_path):
                print(f"Running execution for {n_buckets} buckets, iteration {iteration}")
                results = execute(save_dir=save_dir, n_buckets=n_buckets, i=iteration, statistic="median", similarity_matrices_dir=similarity_matrices_dir)
                bucket_results.append(results)
                all_results[f"{n_buckets}_buckets"] = bucket_results
                with open(all_results_file_path, 'w') as json_file:
                    json.dump(all_results, json_file, indent=4)

        process_rule_similarities(rules_dir=os.path.join(save_dir, "rules"),
                                  dataset_name=f"{dataset_name}",
                                  out_base=os.path.join(save_dir, "rule_similarities"))


if __name__ == '__main__':
    save_dir = "examples/rule_analysis_experiments/mean_meadian_regression/results/concrete_3_breaks/testing"
    run_multiple_executions(save_dir=save_dir, 
                            num_buckets=10, 
                            num_iterations=1,
                            similarity_matrices_dir="examples/XGBRegression/results/concrete_3_breaks/rule_similarities/matrices",
                            dataset_name='concrete_3_breaks')