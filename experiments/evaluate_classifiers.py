import pandas as pd
from EIRegressor import bucketing, replace_nan_median
from EIRegressor.dsgd import DSClassifierMultiQ
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score


def execute():
    # Load dataframe
    data = pd.read_csv("examples/datasets/housing.csv")
    target = "median_house_value"

    # Data Preprocessing
    data["total_bedrooms"].fillna(data["total_bedrooms"].median(), inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    data = data[data[target].notna()]

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(target, axis=1).values,
        data[target].values,
        test_size=0.33,
        random_state=42,
    )
    (buckets_train, bins) = bucketing(y_train, bins=3, type="quantile")

    # If dataset has NaN values, replace them with the median of the column
    medians = replace_nan_median(X_train)
    replace_nan_median(X_test, medians)

    # Create a new set of buckets for the test set with same bins as the train set
    bins[0] = min(y_test) - 1
    bins[-1] = max(y_test) + 1
    y_test = pd.cut(y_test, bins, labels=False)

    DS_classifier = DSClassifierMultiQ(
        num_classes=3,
        max_iter=4000,
        lossfn="MSE",
        min_dloss=0.0001,
        lr=0.005,
        precompute_rules=True,
        force_precompute=True,
        device="cuda",
    )
    DS_classifier.fit(
        X_train,
        buckets_train,
        add_single_rules=True,
        single_rules_breaks=3,
        add_multi_rules=True,
        column_names=data.drop(target, axis=1).columns,
    )
    y_pred = DS_classifier.predict(X_test)

    # Evaluate the model
    f1_macro = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"DS F1 macro: {f1_macro}")
    print(f"DS Accuracy: {acc}")

    DT_classifier = DecisionTreeClassifier(random_state=42)
    DT_classifier.fit(X_train, buckets_train)
    y_pred = DT_classifier.predict(X_test)

    # Evaluate the model
    f1_macro = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    print(f"DT F1 macro: {f1_macro}")
    print(f"DT Accuracy: {acc}")
