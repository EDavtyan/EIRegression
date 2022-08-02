
from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


def execute(n_buckets=3, bucketing_method="quantile"):
   # Load dataframe
    data = pd.read_csv("examples/datasets/movies.csv")
    target = "gross"

    # Data Clean
    data.drop(['movie_title', 'color', 'director_name', 'actor_1_name',
               'actor_2_name', 'actor_3_name', 'language', 'country', 'content_rating', 'aspect_ratio'], axis=1, inplace=True)
    data = data[data['title_year'] >= 1990]
    data = data[data["num_critic_for_reviews"] >= 5]
    data = data[data["num_voted_users"] >= 5]
    data = data[data["movie_facebook_likes"] >= 5]
    data = data[data[target].notna()]

    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)

    # Creation of EI Regression with Gradient Boosting
    eiReg = EmbeddedInterpreter(GradientBoostingRegressor,
                                n_buckets=n_buckets,
                                bucketing_method=bucketing_method,
                                reg_args={"loss": "absolute_error"},
                                max_iter=4000, lossfn="MSE",
                                min_dloss=0.0001, lr=0.005, precompute_rules=True)
    eiReg.fit(X_train, y_train,
              reg_args={},
              add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
              column_names=data.drop(target, axis=1).columns)
    y_pred = eiReg.predict(X_test)

    print("R2: ", r2_score(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    eiReg.print_most_important_rules()

    results = {"R2": r2_score(y_test, y_pred),
               "MAE": mean_absolute_error(y_test, y_pred)}
    eiReg.rules_to_txt("examples/results/movies_results.txt", results=results)
