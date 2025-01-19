# coding=utf-8
import random

import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import torch

from .dsgd.DSClassifierMultiQ import DSClassifierMultiQ
from .dsgd.DSRule import DSRule
from .model_optimizer import ModelOptimizer
from .nanReplace import replace_nan_median
from .bucketing import bucketing


class EmbeddedInterpreter():
    """
    Implementation of Embedded interpreter regression based on DS model
    """

    def __init__(self, regressor=None, model_optimizer=None, model_preprocessor=None, n_buckets=3,
                 bucketing_method="quantile",
                 reg_default_args={}, reg_hp_args={}, hp_grids=None, statistic=None, **cla_kwargs):
        """
        Initialize the EmbeddedInterpreter with new parameters for fine-tuning.
        :param hp_grids: List of hyperparameter grids for each bucket's regressor.
        :param optimizer_settings: Settings for the ModelOptimizer.
        """
        self.n_buckets = n_buckets
        self.bins = []
        self.bucketing_method = bucketing_method
        self.y_dtype = None
        self.training_medians = None
        self.classifier = DSClassifierMultiQ(num_classes=n_buckets, **cla_kwargs)
        if not statistic:
            self.regressors = [regressor(**reg_default_args) for _ in range(n_buckets)]
            self.hp_grids = hp_grids or [reg_hp_args for _ in
                                         range(n_buckets)]  # Default to empty grids if none provided
            self.optimizer = model_optimizer
            self.preprocessor = model_preprocessor
        self.statistic = statistic

    def get_nearest_fitted_regressors(self, i):
        """
        Finds the nearest fitted regressors to bucket i.

        :param i: Bucket index
        :return: List of nearest fitted regressors
        """
        fitted_indices = [j for j, reg in enumerate(self.regressors) if reg is not None]
        if not fitted_indices:
            return None  # No fitted regressors available

        # Calculate distances from bucket i to all fitted buckets
        distances = {j: abs(j - i) for j in fitted_indices}
        min_distance = min(distances.values())

        # Find all regressors at the minimum distance
        nearest_indices = [j for j, d in distances.items() if d == min_distance]

        # Randomly select one of the nearest regressors
        chosen_index = random.choice(nearest_indices)
        return self.regressors[chosen_index]


    def fit(self, X_train, y_train, **cla_kwargs):
        """
        Fit the embedded interpreter model.
        """
        self.y_dtype = y_train.dtype
        if self.bins == []:
            (buckets, bins) = bucketing(
                labels=y_train, features=X_train, bins=self.n_buckets, type=self.bucketing_method)
            self.bins = bins  # To test classifier later
        else:
            buckets = pd.cut(y_train, self.bins)

        # Print bin statistics
        print("\nBin Statistics:")
        print("-" * 50)
        bucket_counts = pd.value_counts(buckets, sort=False)
        for i, count in enumerate(bucket_counts):
            bin_start = self.bins[i] if i == 0 else self.bins[i]
            bin_end = self.bins[i + 1]
            print(f"Bin {i}: Range [{bin_start:.3f}, {bin_end:.3f}], Count: {count}")
        print("-" * 50)

        self.classifier.fit(X_train, buckets, **cla_kwargs)
        pred_bucket = self.classifier.predict(X_train)

        # Print predicted bucket statistics
        print("\nPredicted Bucket Statistics:")
        print("-" * 50)
        pred_counts = pd.value_counts(pred_bucket, sort=False)
        for i in range(self.n_buckets):
            count = pred_counts.get(i, 0)  # Use get to handle cases where a bucket might be empty
            print(f"Predicted Bucket {i}: Count: {count}")
        print("-" * 50)

        self.training_medians = replace_nan_median(X_train)

        if not self.statistic:
            for i in range(self.n_buckets):
                # Get current bucket's data
                current_bucket_X = X_train[pred_bucket == i]
                current_bucket_y = y_train[pred_bucket == i]

                # Initialize bucket_X and bucket_y as the current bucket's data
                bucket_X = current_bucket_X
                bucket_y = current_bucket_y

                # If there are no data points in the current bucket, fall back to true buckets
                if len(bucket_X) == 0:
                    print(f"\nWarning: Predicted bucket {i} is empty, falling back to true bucket")
                    current_bucket_X = X_train[buckets == i]
                    current_bucket_y = y_train[buckets == i]
                    bucket_X = current_bucket_X
                    bucket_y = current_bucket_y
                    print(f"True bucket {i} size: {len(bucket_X)}")

                # Convert to DataFrame if necessary
                if isinstance(bucket_X, np.ndarray):
                    bucket_X = pd.DataFrame(bucket_X)
                if isinstance(bucket_y, np.ndarray):
                    bucket_y = pd.Series(bucket_y)

                # Include the previous bucket (i-1) if it exists
                if i - 1 >= 0:
                    neighbor_X = X_train[pred_bucket == (i - 1)]
                    neighbor_y = y_train[pred_bucket == (i - 1)]
                    if len(neighbor_X) == 0:  # Fallback to true bucket if predicted is empty
                        neighbor_X = X_train[buckets == (i - 1)]
                        neighbor_y = y_train[buckets == (i - 1)]

                    # Convert to DataFrame if necessary
                    if isinstance(neighbor_X, np.ndarray):
                        neighbor_X = pd.DataFrame(neighbor_X)
                    if isinstance(neighbor_y, np.ndarray):
                        neighbor_y = pd.Series(neighbor_y)

                    # Concatenate the data and labels
                    bucket_X = pd.concat([bucket_X, neighbor_X], ignore_index=True)
                    bucket_y = pd.concat([bucket_y, neighbor_y], ignore_index=True)

                # Include the next bucket (i+1) if it exists
                if i + 1 < self.n_buckets:
                    neighbor_X = X_train[pred_bucket == (i + 1)]
                    neighbor_y = y_train[pred_bucket == (i + 1)]
                    if len(neighbor_X) == 0:  # Fallback to true bucket if predicted is empty
                        neighbor_X = X_train[buckets == (i + 1)]
                        neighbor_y = y_train[buckets == (i + 1)]

                    # Convert to DataFrame if necessary
                    if isinstance(neighbor_X, np.ndarray):
                        neighbor_X = pd.DataFrame(neighbor_X)
                    if isinstance(neighbor_y, np.ndarray):
                        neighbor_y = pd.Series(neighbor_y)

                    # Concatenate the data and labels
                    bucket_X = pd.concat([bucket_X, neighbor_X], ignore_index=True)
                    bucket_y = pd.concat([bucket_y, neighbor_y], ignore_index=True)

                # Print final bucket size after including neighbors
                print(f"\nBucket {i} final size (including neighbors): {len(bucket_X)}")

                # Check if the current regressor is None
                if self.regressors[i] is None:
                    nearest_regressor = self.get_nearest_fitted_regressors(i)
                    if nearest_regressor is not None:
                        print(f"\nUsing nearest fitted regressor for bucket {i}")
                        self.regressors[i] = clone(
                            nearest_regressor)  # Clone the regressor to avoid modifying the original
                    else:
                        print(f"\nWarning: No fitted regressors available for bucket {i}")
                        continue  # Skip this bucket if no fitted regressors are available

                # Fit the regressor for the current bucket (and neighbors)
                if self.hp_grids[i]:  # Check if there is a grid for the current bucket
                    regressor_pipeline = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('regressor', self.regressors[i])
                    ])
                    optimized_regressor = self.optimizer.optimize(
                        regressor_pipeline, self.hp_grids[i], bucket_X, bucket_y,
                        scoring='r2',  # Set scoring to a regression metric
                        cv=5  # Or another value, possibly passed through optimizer_settings
                    )
                    self.regressors[i] = optimized_regressor
                else:
                    self.regressors[i].fit(bucket_X, bucket_y)

    def predict(self, X_test, return_buckets=False):
        """
        Predict the classes for the feature vectors
        :param X: Feature vectors
        :param return_buckets: If true, it return buckets assigned to data
        :return: Value predicted for each feature vector. If return_buckets is true, it returns the buckets assigned to data
        """
        buck_pred = self.classifier.predict(X_test)
        y_pred = np.zeros(buck_pred.shape, dtype=self.y_dtype)

        replace_nan_median(X_test, self.training_medians)
        if not self.statistic:
            for i in range(self.n_buckets):
                if not (buck_pred == i).any():
                    continue
                y_pred[buck_pred == i] = self.regressors[i].predict(X_test[buck_pred == i])
        else:
            for i in range(self.n_buckets):
                if not (buck_pred == i).any():
                    continue
                # Use pre-calculated statistics instead of recalculating
                y_pred[buck_pred == i] = self.bucket_statistics[i]

        if return_buckets:
            return buck_pred, y_pred
        return y_pred

    def calculate_bucket_statistics(self, bucket_y, stat_type='median'):
        """
        Calculates median or mean of the targets within a bucket based on the specified statistic type.
        :param bucket_y: Target values for the bucket
        :param stat_type: Type of statistic to compute ('median' or 'mean')
        :return: Computed statistic of bucket_y
        """
        if stat_type == 'median':
            return np.median(bucket_y)
        elif stat_type == 'mean':
            return np.mean(bucket_y)
        else:
            raise ValueError("Invalid stat_type. Use 'median' or 'mean'.")

    def get_bins(self):
        """
        Returns the bins used for bucketing the data
        """
        return self.bins

    def set_bins(self, bins):
        """
        Sets the bins used for bucketing the data
        :param bins: Array of bins
        """
        self.bins = bins

    def predict_proba(self, X):
        """
        Predict the score of belogning to all classes
        :param X: Feature vector
        :return: Class scores for each feature vector
        """
        return self.classifier.predict_proba(X)

    def predict_explain(self, X):
        """
        Predict the score of belogning to each class and give an explanation of that decision
        :param x: A single Feature vectors
        :return: Class scores for each feature vector and a explanation of the decision
        """
        return self.classifier.predict_explain(X)

    def add_rule(self, rule, caption="", m_sing=None, m_uncert=None):
        """
        Adds a rule to the model. If no masses are provided, random masses will be used.
        :param rule: lambda or callable, used as the predicate of the rule
        :param caption: Description of the rule
        :param m_sing: [optional] masses for singletons
        :param m_uncert: [optional] mass for uncertainty
        """
        self.classifier.model.add_rule(DSRule(rule, caption), m_sing, m_uncert)

    def find_most_important_rules(self, classes=None, threshold=0.2):
        """
        Shows the most contributive rules for the classes specified
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :return: A list containing the information about most important rules
        """
        return self.classifier.model.find_most_important_rules(classes=classes, threshold=threshold)

    def print_most_important_rules(self, classes=None, threshold=0.2):
        """
        Prints the most contributive rules for the classes specified
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :return:
        """
        self.classifier.model.print_most_important_rules(
            classes=classes, threshold=threshold)

    def assign_buckets(self, y_values):
        """
        Assigns buckets to the provided y_values based on the bins determined during training.
        :param y_values: Target values to assign buckets to
        :return: Assigned bucket indices
        """
        min_value, max_value = y_values.min(), y_values.max()
        extended_bins = [min(min_value, self.bins[0])] + list(self.bins[1:-1]) + [max(max_value, self.bins[-1])]
        return pd.cut(y_values, bins=extended_bins, labels=False, include_lowest=True)

    def evaluate_classifier(self, X_test, y_test):
        """
        Evaluates the classifier using the test data
        :param X_test: Features for test
        :param y_test: Labels of features
        :return: Accuracy score, F1 macro score, and confusion matrix
        """
        y_test_buckets = self.assign_buckets(y_test)
        y_pred = self.classifier.predict(X_test)
        f1_macro = f1_score(y_test_buckets, y_pred, average='macro')
        acc = accuracy_score(y_test_buckets, y_pred)
        cm = confusion_matrix(y_test_buckets, y_pred)
        return acc, f1_macro, cm

    def rules_to_txt(self, filename, classes=None, threshold=0.2, results={}):
        """
        Write the most contributive rules for the classes specified in an output file
        :param filename: Output file name
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :param results: Dictionary with the results to print in txt
        :return:
        """
        rules = self.classifier.model.find_most_important_rules(
            classes=classes, threshold=threshold)
        with open(filename, 'w') as file:
            for r in results:
                file.write(r + ": " + str(results[r]) + "\n\n")
            file.write(f"Most important rules\n-----------------------------\n")
            for key, rules_list in rules.items():
                file.write(f"\n---{key}---\n")
                for rule in rules_list:
                    file.write(
                        f"rule{rule[1]}: {rule[2]}\nprobabilities_array:{rule[4]}\n\n")

    def get_top_uncertainties(self, classes=None, threshold=0.2, top=5):
        rules = self.classifier.model.find_most_important_rules(
            classes=classes, threshold=threshold)
        results = {}

        for key, rules_list in rules.items():
            all_scores = []

            for rule in rules_list:
                uncertainty_score = rule[4][-1]
                all_scores.append(float(uncertainty_score))

            if len(all_scores) < top:
                top_scores = sorted(all_scores, reverse=False)
            else:
                top_scores = sorted(all_scores, reverse=False)[:top]

            results[f"class{key}"] = top_scores

        return results

    def compute_similarity(self, x, y_true, threshold=0.2):
        """
        Compute the similarity between the activated rules for input x and the most important rules for the actual class y_true.
        Similarity is defined as the proportion of the intersection between these two rule sets over the union.
        :param x: Input sample (feature vector)
        :param y_true: Actual class label (integer index)
        :param threshold: Threshold for important rules
        :return: Similarity score
        """
        # Get the activated rules for x
        rules, preds = self.classifier.model.get_rules_by_instance(x)
        # Represent the activated rules as a set of their string representations
        activated_rules_set = set([str(pred) for pred in preds])

        # Get the most important rules for y_true
        important_rules_dict = self.classifier.model.find_most_important_rules(classes=[y_true], threshold=threshold)
        important_rules_list = important_rules_dict.get(y_true, [])
        # The important rules are stored as tuples; the third element is the rule's caption (string representation)
        important_rules_set = set([rule_info[2] for rule_info in important_rules_list])

        # Compute the intersection and union of the two rule sets
        intersection = activated_rules_set & important_rules_set
        # union = activated_rules_set | important_rules_set
        union = important_rules_set

        # Compute the similarity as the proportion of intersection over union
        if len(union) == 0:
            similarity = 0.0
        else:
            similarity = len(intersection) / len(union)

        return similarity
