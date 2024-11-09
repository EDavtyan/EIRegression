import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

os.chdir('/home/davtyan.edd/projects/EIRegression/')

from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter
from EIRegressor.utils import compute_weighted_accuracy


def compute_similarity_metrics(eiReg, X, y, y_true, n_buckets, threshold=0.2):
    # Assign buckets to y and y_true 
    y_pred_buckets = eiReg.assign_buckets(y)
    y_true_buckets = eiReg.assign_buckets(y_true)

    # Initialize similarity matrix and counters
    similarity_matrix = np.zeros((n_buckets, n_buckets))
    counts = np.zeros(n_buckets)
    pred_counts = np.zeros(n_buckets)

    # List to store similarities with actual classes
    similarity_scores = []

    # For each test sample
    for x_sample, y_pred, y_actual in zip(X, y_pred_buckets, y_true_buckets):
        y_pred = int(y_pred)
        y_actual = int(y_actual)

        # Increment the count for the actual bucket
        counts[y_actual] += 1

        # Increment the count for the predicted bucket
        pred_counts[y_pred] += 1


        similarity = 1 if y_pred == y_actual else eiReg.compute_similarity(x_sample, y_actual, threshold=threshold, include_rule_coverage=True)
        # similarity = eiReg.compute_similarity(x_sample, y_actual, threshold=threshold, include_rule_coverage=True)
        similarity_matrix[y_actual, y_pred] += similarity

        similarity_scores.append(similarity)

    # Normalize the similarity matrix by the number of records for each predicted bucket
    for i in range(n_buckets):
        for j in range(n_buckets):
            if pred_counts[j] > 0:
                similarity_matrix[i, j] /= pred_counts[j]

    # Compute average similarity
    average_similarity = np.mean(similarity_scores) if similarity_scores else 0.0

    return average_similarity, similarity_matrix

def plot_similarity_heatmap(average_similarity_matrix, n_buckets, save_dir):
    average_similarity_matrix_plot_dir = os.path.join(save_dir, "rule_similarities_V3_1/heat_maps")

    plt.figure(figsize=(10, 8))
    sns.heatmap(average_similarity_matrix, annot=True, fmt=".2f", cmap='viridis')
    plt.title(f"Average Similarity Matrix for {n_buckets} Buckets")
    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Buckets")
    plt.tight_layout()
    os.makedirs(average_similarity_matrix_plot_dir, exist_ok=True)
    plt.savefig(os.path.join(average_similarity_matrix_plot_dir, f"average_similarity_matrix_{n_buckets}_buckets.png"))
    plt.close()

def execute(save_dir, n_buckets=3, i=None, bucketing_method="quantile", statistic="median"):
    # Load dataframe
    data = pd.read_csv("examples/datasets/concrete_data.csv")
    target = "concrete_compressive_strength"
    data = data[data[target].notna()]

    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)

    # Create the EI Regression model
    eiReg = EmbeddedInterpreter(
        n_buckets=n_buckets,
        bucketing_method=bucketing_method,
        statistic=statistic,
        max_iter=4000,
        lossfn="MSE",
        min_dloss=0.0001,
        lr=0.005,
        precompute_rules=True,
        force_precompute=True,
        device="cpu"
    )

    # Fit the model
    eiReg.fit(X_train, y_train,
              add_single_rules=True, single_rules_breaks=3, add_mult_rules=True,
              column_names=data.drop(target, axis=1).columns)

    # Get predictions and buckets
    buck_pred, y_pred = eiReg.predict(X_test, return_buckets=True)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    acc, f1, cm = eiReg.evaluate_classifier(X_test, y_test)

    # Compute similarity scores and similarity matrix
    average_similarity, similarity_matrix = compute_similarity_metrics(eiReg, X_test, y_pred, y_test, n_buckets)

    top_uncertainties = eiReg.get_top_uncertainties()

    results = {
        "MSE": mse,
        "Accuracy": acc,
        "F1": f1,
        "Confusion Matrix": cm.tolist(),
        "Average Similarity": average_similarity,
        "Similarity Matrix": similarity_matrix.tolist(),
        "Uncertainties": top_uncertainties,
        # Temporarily saving the below items for computing the weighted accuracy
        # "y_test": y_test.tolist(),
        # "buck_pred": buck_pred.tolist(),
        # "bins": eiReg.get_bins().tolist()
    }

    # Save the rules
    results_dir = os.path.join(save_dir, "rules")
    os.makedirs(results_dir, exist_ok=True)
    eiReg.rules_to_txt(os.path.join(results_dir, f"rule_results_{n_buckets}_buckets_{i}_iterations.txt"),
                       results=results)

    return results

def run_experiments_for_buckets(save_dir, n_buckets, num_iterations, bucket_results):
    for iteration in range(1, num_iterations + 1):
        # Construct the expected path for the results of this iteration
        expected_result_path = os.path.join(save_dir, "rules", f"rule_results_{n_buckets}_buckets_{iteration}_iterations.txt")

        # Check if this experiment's results already exist
        if not os.path.exists(expected_result_path):
            print(f"Running execution for {n_buckets} buckets, iteration {iteration}")
            results = execute(save_dir=save_dir, n_buckets=n_buckets, i=iteration, statistic="median")
            bucket_results.append(results)
            # print("results", results)
    return bucket_results

def update_results_with_weighted_accuracy(save_dir, bucket_results, n_buckets):
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

def compute_and_save_average_similarity_matrix(save_dir, bucket_results, n_buckets):
    similarity_matrices = []
    for result in bucket_results:
        sim_matrix = np.array(result['Similarity Matrix'])
        similarity_matrices.append(sim_matrix)

    average_similarity_matrix = np.mean(similarity_matrices, axis=0)

    # Save the average similarity matrix
    average_similarity_matrix_path = os.path.join(save_dir, "rule_similarities_V3_1/matricies", f"average_similarity_matrix_{n_buckets}_buckets.npy")
    os.makedirs(os.path.dirname(average_similarity_matrix_path), exist_ok=True)
    np.save(average_similarity_matrix_path, average_similarity_matrix)

    # Optionally, save a heatmap of the average similarity matrix
    plot_similarity_heatmap(average_similarity_matrix, n_buckets, save_dir)
