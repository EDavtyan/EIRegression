import os
import sys
import json
import pandas as pd
import warnings

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Append the project path to sys.path and set the working directory
sys.path.append('/home/davtyan.edd/EIRegression/')
os.chdir('/home/davtyan.edd/EIRegression/')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter
from EIRegressor.model_optimizer import ModelOptimizer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# -------------------------------
# Helper function: Datetime feature extraction
# -------------------------------
def extract_datetime_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    return df

# -------------------------------
# Load and preprocess energydata_complete
# -------------------------------
def load_energy_data(dataset_path="examples/datasets/energydata_complete.csv"):
    """
    Loads and preprocesses the energydata_complete dataset.

    Returns:
    - X_train_np (np.ndarray): Training features (NumPy array for classifier).
    - y_train_np (np.ndarray): Training target (NumPy array for classifier).
    - X_test_np (np.ndarray): Testing features (NumPy array for classifier).
    - y_test_np (np.ndarray): Testing target (NumPy array for classifier).
    - X_train_df (pd.DataFrame): Training features (DataFrame for regressor pipeline).
    - X_test_df (pd.DataFrame): Testing features (DataFrame for regressor pipeline).
    - feature_columns (list): List of feature column names.
    - model_preprocessor (sklearn.pipeline.Pipeline): Preprocessor for regressor pipeline.
    """
    # Load CSV data
    df = pd.read_csv(dataset_path)

    # Extract datetime features
    df = extract_datetime_features(df)

    # Drop the original date column as its information is now encoded
    df = df.drop(columns=["date"])

    # Set the target column
    target = "Appliances"

    # Separate target from features
    y = df[target]
    X = df.drop(columns=[target])

    # Define feature groups for unified preprocessor:
    # We'll treat the newly created datetime features as categorical (to be one-hot encoded)
    datetime_features = ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'is_weekend']
    continuous_features = [col for col in X.columns if col not in datetime_features]

    # Convert feature names to indices for ColumnTransformer
    datetime_indices = [X.columns.get_loc(col) for col in datetime_features]
    continuous_indices = [X.columns.get_loc(col) for col in continuous_features]

    # Create unified preprocessor using column indices (DataFrame input is preserved)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_indices),
            ('cat', OneHotEncoder(handle_unknown='ignore'), datetime_indices)
        ]
    )
    
    # Create a pipeline that holds the preprocessor
    model_preprocessor = Pipeline(steps=[('preprocessor', preprocessor)])

    # For feature naming, list the original feature names.
    feature_columns = list(X.columns)

    # Split the data into training and testing sets (retain DataFrame format)
    X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # Convert to NumPy arrays for the classifier (which uses NumPy slicing)
    X_train_np = X_train_df.values
    X_test_np = X_test_df.values
    y_train_np = y_train_series.values
    y_test_np = y_test_series.values

    # Return both DataFrame and NumPy array versions
    return (
        X_train_np,  # NumPy array for classifier
        y_train_np,
        X_test_np,   # NumPy array for classifier
        y_test_np,
        X_train_df,  # DataFrame for regressor pipeline
        X_test_df,
        feature_columns, 
        model_preprocessor
    )

# -------------------------------
# Execution function for a single experiment with XGBoost
# -------------------------------
def execute(save_dir, n_buckets=3, i=None, bucketing_method="quantile", single_rules_breaks=3):
    """
    Executes a single XGBoost regression experiment on energydata_complete
    using a unified preprocessor for both NN and tree-based models.

    Parameters:
    - save_dir (str): Directory to save the results.
    - n_buckets (int): Number of buckets for bucketing.
    - i (int): Iteration number.
    - bucketing_method (str): Method for bucketing.
    - single_rules_breaks (int): Number of breaks for single rules.

    Returns:
    - dict: Dictionary containing evaluation metrics and uncertainties.
    """
    # Load data in both DataFrame and NumPy formats
    X_train_np, y_train_np, X_test_np, y_test_np, X_train_df, X_test_df, feature_columns, model_preprocessor = load_energy_data()

    # Define hyperparameter grid for XGBoost
    regressor_hp_grid = {
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__n_estimators': [50, 100, 150],
        'regressor__max_depth': [3, 5, 7],
        'regressor__colsample_bytree': [0.7, 0.8, 0.9],
        'regressor__gamma': [0, 0.1, 0.2],
        'regressor__reg_alpha': [0, 0.1, 0.5],
        'regressor__reg_lambda': [0.5, 1, 1.5],
        'regressor__n_jobs': [-1]
    }

    # Default hyperparameters for XGBoost regressor
    regressor_default_args = {
        "learning_rate": 0.1,
        "n_estimators": 100,
        "max_depth": 5,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "n_jobs": -1,
        "objective": "reg:squarederror"
    }

    # Initialize the model optimizer with grid search
    regressor_optimizer = ModelOptimizer(search_method="grid")

    # Initialize the Embedded Interpreter with the unified preprocessor,
    # using XGBRegressor for the tree-based branch.
    eiReg = EmbeddedInterpreter(
        regressor=XGBRegressor,
        model_optimizer=regressor_optimizer,
        model_preprocessor=model_preprocessor,
        n_buckets=n_buckets,
        bucketing_method=bucketing_method,
        reg_default_args=regressor_default_args,
        reg_hp_args=regressor_hp_grid,
        max_iter=4000,
        lossfn="MSE",
        min_dloss=0.0001,
        lr=0.005,
        precompute_rules=True,
        force_precompute=True,
        device="cuda",
        verbose=True
    )

    # Fit the model (pass NumPy arrays for the classifier)
    eiReg.fit(
        X_train_np,
        y_train_np,
        add_single_rules=True,
        single_rules_breaks=single_rules_breaks,
        add_multi_rules=True,
        column_names=feature_columns
    )

    # Predict on test data using NumPy arrays
    y_pred = eiReg.predict(X_test_np)

    # Evaluate classifier using NumPy arrays
    acc, f1, cm = eiReg.evaluate_classifier(X_test_np, y_test_np)

    # Calculate evaluation metrics
    r2 = r2_score(y_test_np, y_pred)
    mae = mean_absolute_error(y_test_np, y_pred)
    mse = mean_squared_error(y_test_np, y_pred)

    # Get top uncertainties from the rules
    top_uncertainties = eiReg.get_top_uncertainties()

    # Compile results into a dictionary
    results = {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "Accuracy": acc,
        "F1": f1,
        "Confusion Matrix": cm.tolist(),
        "Uncertainties": top_uncertainties
    }

    # Save the rules and results in a specified directory
    save_results = os.path.join(save_dir, "rules")
    os.makedirs(save_results, exist_ok=True)
    rule_filename = f"rule_results_{n_buckets}_buckets_{i}_iterations.txt" if i is not None else f"rule_results_{n_buckets}_buckets.txt"
    eiReg.rules_to_txt(
        os.path.join(save_results, rule_filename),
        results=results
    )

    return results

# -------------------------------
# Run multiple executions for experiments with XGBoost
# -------------------------------
def run_multiple_executions(save_dir, num_buckets, num_iterations, dataset_name='energydata_complete', single_rules_breaks=3):
    """
    Runs multiple XGBoost regression experiments on energydata_complete across
    different bucket numbers and iterations.

    Parameters:
    - save_dir (str): Directory to save all results.
    - num_buckets (int): Maximum number of buckets to iterate through.
    - num_iterations (int): Number of iterations per bucket.
    - dataset_name (str): Name of the dataset (used for naming result files).
    - single_rules_breaks (int): Number of breaks for single rules.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results_file_path = os.path.join(
        save_dir, f"{dataset_name}_results_{num_buckets}_buckets_{num_iterations}_iterations.json"
    )

    all_results = {}
    if os.path.exists(all_results_file_path):
        with open(all_results_file_path, 'r') as json_file:
            all_results = json.load(json_file)

    for n_buckets in range(1, num_buckets + 1):
        bucket_key = f"{n_buckets}_buckets"
        bucket_results = all_results.get(bucket_key, [])

        for iteration in range(1, num_iterations + 1):
            expected_result_path = os.path.join(
                save_dir, "rules", f"rule_results_{n_buckets}_buckets_{iteration}_iterations.txt"
            )
            if not os.path.exists(expected_result_path):
                print(f"Running execution for {n_buckets} bucket(s), iteration {iteration}")
                results = execute(
                    save_dir=save_dir,
                    n_buckets=n_buckets,
                    i=iteration,
                    single_rules_breaks=single_rules_breaks
                )
                bucket_results.append(results)
                all_results[bucket_key] = bucket_results

                with open(all_results_file_path, 'w') as json_file:
                    json.dump(all_results, json_file, indent=4)

if __name__ == '__main__':
    dataset_name = "energydata_complete"
    
    save_dir = os.path.join(
        "examples/XGBRegression/results/",
        dataset_name
    )

    run_multiple_executions(
        save_dir=save_dir,
        num_buckets=10,
        num_iterations=10,
        dataset_name=dataset_name
    )

    print("\n" + "="*47)
    print("✨🎉   Thank you for using this program!   🎉✨")
    print("        🚀 Program executed successfully 🚀")
    print("="*47 + "\n")