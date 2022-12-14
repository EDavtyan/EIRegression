
import tarfile
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

from EIRegressor import bucketing, replace_nan_median

name = "DT concrete"
def execute():
   # Load dataframe
    data = pd.read_csv("examples/datasets/concrete_data.csv")
    target = "concrete_compressive_strength"
    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)
    
    data[target] = bucketing(
        data[target], type="quantile", bins=3)[0]

    # Data Split
    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    medians = replace_nan_median(X_train)
    replace_nan_median(X_test, medians)

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
                                    max_depth=2,
                                    impurity=False

                                    )
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render(f'./experiments/figures/{name}', view=True)
    plt.show()
