import os, sys
import numpy as np
import pandas as pd
import json

sys.path.append('/home/davtyan.edd/projects/EIRegression/')
os.chdir('/home/davtyan.edd/projects/EIRegression/')

from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter
from EIRegressor.utils import compute_weighted_accuracy
from rule_analysis.compare_rules import process_rule_similarities
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix


def execute(save_dir, n_buckets=3, i=None, bucketing_method="quantile", statistic="median"):
    train_data = pd.read_csv(
        "examples/datasets/bank32NH/bank32nh.data",
        delim_whitespace=True, header=None)
    test_data = pd.read_csv(
        "examples/datasets/bank32NH/bank32nh.test",
        delim_whitespace=True, header=None)

    # Load domain file for column names
    # domain_data = pd.read_csv("/Users/eddavtyan/Documents/XAI/Projects/EIRegression/examples/datasets/bank32NH/bank32nh.domain", delim_whitespace=True, header=None)
    domain_data = pd.read_csv(
        "examples/datasets/bank32NH/bank32nh.domain",
        delim_whitespace=True, header=None)
    column_names = domain_data.iloc[:, 0].apply(lambda x: x.split()[0]).tolist()

    # Apply column names to the train and test dataframes
    train_data.columns = column_names
    test_data.columns = column_names

    # Set the target column (assuming 'b2call' for this example)
    target = "rej"

    # Data Preprocessing
    train_data = train_data.apply(pd.to_numeric, errors='ignore')
    test_data = test_data.apply(pd.to_numeric, errors='ignore')

    train_data = pd.get_dummies(train_data, drop_first=True)
    test_data = pd.get_dummies(test_data, drop_first=True)

    train_data = train_data[train_data[target].notna()]
    test_data = test_data[test_data[target].notna()]

    X_train, y_train = train_data.drop(target, axis=1).values, train_data[target].values
    X_test, y_test = test_data.drop(target, axis=1).values, test_data[target].values

    # Creation of EI Regression with XGBoost
    eiReg = EmbeddedInterpreter(n_buckets=n_buckets,
                                bucketing_method=bucketing_method,
                                statistic=statistic,
                                max_iter=4000, lossfn="MSE",
                                min_dloss=0.0001, lr=0.005, precompute_rules=True,
                                force_precompute=True, device="cpu")

    eiReg.fit(X_train, y_train,
              add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
              column_names=train_data.drop(target, axis=1).columns)
    
    # Get predictions and buckets
    buck_pred, y_pred = eiReg.predict(X_test, return_buckets=True)
    
    
    # r2 = r2_score(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    acc, f1, cm = eiReg.evaluate_classifier(X_test, y_test)

    top_uncertainties = eiReg.get_top_uncertainties()

    results = {
            #    "R2": r2,
            #    "MAE": mae,
               "MSE": mse,
               "Accuracy": acc,
               "F1": f1,
               "Confusion Matrix": cm.tolist(),
               "Uncertainties": top_uncertainties,

               # temporarely saving the below items for computing the weighted accuracy
               "y_test": y_test.tolist(),
               "buck_pred": buck_pred.tolist(),
               "bins": eiReg.get_bins().tolist()
               }
    
    # Save the rules
    results_dir = os.path.join(save_dir, "rules")
    os.makedirs(results_dir, exist_ok=True)
    eiReg.rules_to_txt(os.path.join(results_dir, f"rule_results_{n_buckets}_buckets_{i}_iterations.txt"),
                       results=results)

    return results


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

        for iteration in range(1, num_iterations + 1):
            # Construct the expected path for the results of this iteration
            expected_result_path = os.path.join(save_dir, "rules", f"rule_results_{n_buckets}_buckets_{iteration}_iterations.txt")

            # Check if this experiment's results already exist
            if not os.path.exists(expected_result_path):
                print(f"Running execution for {n_buckets} buckets, iteration {iteration}")
                results = execute(save_dir=save_dir, n_buckets=n_buckets, i=iteration, statistic="median")
                bucket_results.append(results)
                all_results[f"{n_buckets}_buckets"] = bucket_results
                with open(all_results_file_path, 'w') as json_file:
                    json.dump(all_results, json_file, indent=4)

        # Process rule similarities after all iterations for the current number of buckets
        process_rule_similarities(rules_dir=os.path.join(save_dir, "rules"),
                                  dataset_name=f"{dataset_name}",
                                  out_base=os.path.join(save_dir, "rule_similarities"))
        
        # Now recompute weighted_accuracy for each iteration with the aggregated rule similarity matrices
        similarity_matrices_dir_iter = os.path.join(save_dir, "rule_similarities", "matrices")
        for idx, result in enumerate(bucket_results):
            y_test = np.array(result['y_test'])
            buck_pred = np.array(result['buck_pred'])
            bins = result['bins']
            # Compute weighted accuracy
            weighted_accuracy = compute_weighted_accuracy(
                actual_values=y_test,
                predicted_buckets=buck_pred,
                bins=bins,
                n_buckets=n_buckets,
                similarity_matrices_dir=similarity_matrices_dir_iter
            )
            # Update result with weighted accuracy
            result['Weighted Accuracy'] = weighted_accuracy
            # Remove unnecessary data
            del result['y_test']
            del result['buck_pred']
            del result['bins']
            bucket_results[idx] = result

        # Update the results file after computing weighted accuracy
        all_results[f"{n_buckets}_buckets"] = bucket_results
        with open(all_results_file_path, 'w') as json_file:
            json.dump(all_results, json_file, indent=4)


if __name__ == '__main__':
    dataset_name="bank32NH_3_breaks"

    save_dir = os.path.join("examples/rule_analysis_experiments/results", dataset_name)
    run_multiple_executions(save_dir=save_dir,
                            num_buckets=10,
                            num_iterations=25,
                            dataset_name=dataset_name)