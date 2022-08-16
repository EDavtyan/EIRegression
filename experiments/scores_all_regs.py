import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from pyfume import pyFUME

from EIRegressor import replace_nan_median


def execute():
    MAE_GB = []
    R2_GB = []
    MAE_RFOREST = []
    R2_RFOREST = []
    MAE_MLP = []
    R2_MLP = []
    MAE_LINEAR = []
    R2_LINEAR = []
    MAE_PYFUME = []
    R2_PYFUME = []

    # Load dataframe
    data = pd.read_csv("experiments/datasets/insurance.csv")
    target = "charges"

    # Data Preprocessing
    data = data.apply(pd.to_numeric, errors='ignore')
    data = pd.get_dummies(data, drop_first=True)
    data = data[data[target].notna()]

    # Reorder to pyFUME
    cols = data.columns.tolist()
    cols = cols[4:] + cols[:4]
    data = data[cols]
    data.columns = data.columns.str.replace(" ", "_")

    print(data.columns)
    # Data Split
    X, y = data.drop(target, axis=1).values, data[target].values

    for i in range(50):
        print(i+1)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        medians = replace_nan_median(X_train)
        replace_nan_median(X_test, medians)

        gb = GradientBoostingRegressor(
            random_state=42, loss='absolute_error', n_estimators=300).fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        print(f"r2 of GradientBoostingRegressor is {r2_score(y_test, y_pred)}")
        print(
            f"MAE of GradientBoostingRegressor is {mean_absolute_error(y_test, y_pred)}")

        R2_GB += [r2_score(y_test, y_pred)]
        MAE_GB += [mean_absolute_error(y_test, y_pred)]

        rfRegressor = RandomForestRegressor(
            random_state=42, n_estimators=300).fit(X_train, y_train)
        y_pred = rfRegressor.predict(X_test)
        print(f"r2 of randomForestRegressor is {r2_score(y_test, y_pred)}")
        print(
            f"MAE of randomForestRegressor is {mean_absolute_error(y_test, y_pred)}")

        R2_RFOREST += [r2_score(y_test, y_pred)]
        MAE_RFOREST += [mean_absolute_error(y_test, y_pred)]

        sc = StandardScaler()
        xtrain = sc.fit_transform(X_train)
        xtest = sc.transform(X_test)

        mlp = MLPRegressor(solver="lbfgs").fit(xtrain, y_train)
        y_pred = mlp.predict(xtest)
        print(f"r2 of MLPRegressor is {r2_score(y_test, y_pred)}")
        print(f"MAE of MLPRegressor is {mean_absolute_error(y_test, y_pred)}")

        R2_MLP += [r2_score(y_test, y_pred)]
        MAE_MLP += [mean_absolute_error(y_test, y_pred)]

        linearRegressor = LinearRegression().fit(X_train, y_train)
        y_pred = linearRegressor.predict(X_test)
        print(f"r2 of linearRegressor is {r2_score(y_test, y_pred)}")
        print(
            f"MAE of linearRegressor is {mean_absolute_error(y_test, y_pred)}")

        R2_LINEAR += [r2_score(y_test, y_pred)]
        MAE_LINEAR += [mean_absolute_error(y_test, y_pred)]

        pyfumeReg = pyFUME(dataframe=data, nr_clus=3,
                           percentage_training=0.75)
        MAE = pyfumeReg.calculate_error(method="MAE")
        R2 = 1 - pyfumeReg.calculate_error(method="MSE")/y_test.var()
        print(f"r2 of FuzzyRulesRegressor is {R2}")
        print(f"MAE of FuzzyRulesRegressor is {MAE}")

        R2_PYFUME += [R2]
        MAE_PYFUME += [MAE]

    print("MAE: ", [MAE_GB, MAE_RFOREST, MAE_MLP, MAE_LINEAR, MAE_PYFUME])
    print("R2: ", [R2_GB, R2_RFOREST, R2_MLP, R2_LINEAR, R2_PYFUME])

    with open("experiments/results/all_regs.txt", 'w') as file:
        file.write(f"All Regressors results\n-----------------------------\n\n")
        file.write(f"""
    MAE GB: {MAE_GB}\n
    MAE RFOREST: {MAE_RFOREST}\n
    MAE MLP: {MAE_MLP}\n
    MAE LINEAR: {MAE_LINEAR}\n
    MAE PYFUME: {MAE_PYFUME}\n
    ALL MAE: {[MAE_GB, MAE_RFOREST, MAE_MLP, MAE_LINEAR, MAE_PYFUME]}\n
        """)
        file.write("\n-----------------------------\n\n")
        file.write(f"""
    R2 GB: {R2_GB}\n
    R2 RFOREST: {R2_RFOREST}\n
    R2 MLP: {R2_MLP}\n
    R2 LINEAR: {R2_LINEAR}\n
    R2 PYFUME: {R2_PYFUME}\n
    ALL R2: {[R2_GB, R2_RFOREST, R2_MLP, R2_LINEAR, R2_PYFUME]}\n
        """)
        file.close()
