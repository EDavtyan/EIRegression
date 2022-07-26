
import tarfile
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

from EIRegressor import bucketing, replace_nan_median


def execute():
    # Load dataframe
    data = pd.read_csv("./examples/datasets/movies.csv")
    target = "gross"

    # Data Clean
    data.drop(['movie_title', 'color', 'director_name', 'actor_1_name',
               'actor_2_name', 'actor_3_name', 'language', 'country', 'content_rating', 'aspect_ratio'], axis=1, inplace=True)
    data = data[data['title_year'] >= 1990]
    data = data[data["num_critic_for_reviews"] >= 5]
    data = data[data["num_voted_users"] >= 5]
    data = data[data["movie_facebook_likes"] >= 5]
    data = data[data[target].notna()]

    data[target] = bucketing(
        data[target], type="quantile", bins=3)[0]

    # Data Split
    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    replace_nan_median(X_train)
    replace_nan_median(X_test)

    # , min_impurity_decrease=0.0189)
    dt = DecisionTreeClassifier(random_state=42)
    # Train model
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))

    dot_data = tree.export_graphviz(dt,
                                    feature_names=data.drop(
                                        target, axis=1).columns,
                                    class_names=[
                                        f"low_{target}", f"medium_{target}", f"high_{target}"],
                                    filled=True, rounded=True,
                                    special_characters=True,
                                    max_depth=3,
                                    impurity=False

                                    )
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render('./experiments/figures/imdb_DT', view=True)
    plt.show()
