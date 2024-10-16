import sys
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter
from EIRegressor.model_optimizer import ModelOptimizer

sys.path.append('/home/davtyan.edd/projects/EIRegression')


def execute(save_dir, n_buckets=3, i=None, bucketing_method="kmeans", statistic="median"):
    # Load dataframe
    data = pd.read_csv("/home/edgar.davtyan/projects/recla_v1/examples/datasets/concrete_data.csv")
    target = "concrete_compressive_strength"

    # Data Preprocessing
    data = data[data[target].notna()]
    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Creation of EI Regression with XGBoost
    eiReg = EmbeddedInterpreter(xgb.XGBRegressor,
                                n_buckets=n_buckets,
                                bucketing_method=bucketing_method,
                                reg_args={'objective': 'reg:squarederror'},
                                max_iter=4000, lossfn="MSE",
                                min_dloss=0.0001, lr=0.005, precompute_rules=True,
                                force_precompute=True, device="cpu")
    eiReg.fit(X_train, y_train,
              add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
              column_names=data.drop(target, axis=1).columns)
    y_pred = eiReg.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_path)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R2: {r2}, MAE: {mae}, MSE: {mse}")
    eiReg.print_most_important_rules()

    results = {"R2": r2, "MAE": mae, "MSE": mse}
    eiReg.rules_to_txt(os.path.join(save_dir, f"rule_results_{n_buckets}_buckets_{i}.txt"), results=results)

    return results


if __name__ == '__main__':
    save_dir = "/home/edgar.davtyan/projects/recla_v1/examples/results"
    os.makedirs(save_dir, exist_ok=True)
    execute(save_dir=save_dir, n_buckets=3, i=1, bucketing_result="quantile")