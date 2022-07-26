import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from pyfume import pyFUME

from EIRegressor import EmbeddedInterpreter, replace_nan_median


def execute():
    MAE_GB_EMB = []
    R2_GB_EMB = []
    MAE_RFOREST_EMB = []
    R2_RFOREST_EMB = []
    MAE_MLP_EMB = []
    R2_MLP_EMB = []
    MAE_LINEAR_EMB = []
    R2_LINEAR_EMB = []

    # Load dataframe
    data = pd.read_csv("./examples/datasets/concrete_data.csv")
    target = "concrete_compressive_strength"

    for i in range(5):
        print(i+1)
        # Data Split
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(target, axis=1).values, data[target].values, test_size=0.33)

        # replace_nan_median(X_train)
        # replace_nan_median(X_test)

        gbrreg = GradientBoostingRegressor
        EIgb = EmbeddedInterpreter(gbrreg,
                                   reg_args={},
                                   n_buckets=3, max_iter=4000, lossfn="MSE",
                                   min_dloss=0.0001, lr=0.005, precompute_rules=True)

        EIgb.fit(X_train, y_train,
                 reg_args={},
                 add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
                 column_names=data.drop(target, axis=1).columns)
        y_pred = EIgb.predict(X_test)

        print("R2 for GradientBoostingEmbedded: ", r2_score(y_test, y_pred))
        print("MAE for GradientBoostingEmbedded: ",
              mean_absolute_error(y_test, y_pred))

        R2_GB_EMB += [r2_score(y_test, y_pred)]
        MAE_GB_EMB += [mean_absolute_error(y_test, y_pred)]

        rfReg = RandomForestRegressor
        EIrf = EmbeddedInterpreter(rfReg,
                                   reg_args={},
                                   n_buckets=3, max_iter=4000, lossfn="MSE",
                                   min_dloss=0.0001, lr=0.005, precompute_rules=True)

        EIrf.fit(X_train, y_train,
                 reg_args={},
                 add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
                 column_names=data.drop(target, axis=1).columns)
        y_pred = EIrf.predict(X_test)

        print("R2 for RandomForestEmbedded: ", r2_score(y_test, y_pred))
        print("MAE for RandomForestEmbedded: ",
              mean_absolute_error(y_test, y_pred))

        R2_RFOREST_EMB += [r2_score(y_test, y_pred)]
        MAE_RFOREST_EMB += [mean_absolute_error(y_test, y_pred)]

        sc = StandardScaler()
        xtrain = sc.fit_transform(X_train)
        xtest = sc.transform(X_test)

        mlpreg = MLPRegressor
        EImlp = EmbeddedInterpreter(mlpreg,
                                    reg_args={"solver": "lbfgs"},
                                    n_buckets=3, max_iter=4000, lossfn="MSE",
                                    min_dloss=0.0001, lr=0.005, precompute_rules=True)

        EImlp.fit(xtrain, y_train,
                  reg_args={},
                  add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
                  column_names=data.drop(target, axis=1).columns)
        y_pred = EImlp.predict(xtest)

        print("R2 for MLPRegressionEmbedded: ", r2_score(y_test, y_pred))
        print("MAE for MLPRegressionEmbedded: ",
              mean_absolute_error(y_test, y_pred))

        R2_MLP_EMB += [r2_score(y_test, y_pred)]
        MAE_MLP_EMB += [mean_absolute_error(y_test, y_pred)]

        lrreg = LinearRegression
        EIlr = EmbeddedInterpreter(lrreg,
                                   reg_args={},
                                   n_buckets=3, max_iter=4000, lossfn="MSE",
                                   min_dloss=0.0001, lr=0.005, precompute_rules=True)

        EIlr.fit(X_train, y_train,
                 reg_args={},
                 add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
                 column_names=data.drop(target, axis=1).columns)
        y_pred = EIlr.predict(X_test)

        print("R2 for LinearRegressionEmbedded: ", r2_score(y_test, y_pred))
        print("MAE for LinearRegressionEmbedded: ",
              mean_absolute_error(y_test, y_pred))

        R2_LINEAR_EMB += [r2_score(y_test, y_pred)]
        MAE_LINEAR_EMB += [mean_absolute_error(y_test, y_pred)]

    print("MAE: ", [MAE_GB_EMB, MAE_RFOREST_EMB,
          MAE_MLP_EMB, MAE_LINEAR_EMB])
    print("R2: ", [R2_GB_EMB, R2_RFOREST_EMB,
          R2_MLP_EMB, R2_LINEAR_EMB])

    EIgb.rules_to_txt("experiments/results/emb_gb_rules.txt")
    EIrf.rules_to_txt("experiments/results/emb_rf_rules.txt")
    EImlp.rules_to_txt("experiments/results/emb_mlp_rules.txt")
    EIlr.rules_to_txt("experiments/results/emb_linear_rules.txt")

    with open("experiments/results/embedded_regs.txt", 'w') as file:
        file.write(
            f"All Regressors Embedded results\n-----------------------------\n\n")
        file.write(f"""
    MAE GB: {MAE_GB_EMB}\n
    MAE RFOREST: {MAE_RFOREST_EMB}\n
    MAE MLP: {MAE_MLP_EMB}\n
    MAE LINEAR: {MAE_LINEAR_EMB}\n
    ALL MAE: {[MAE_GB_EMB, MAE_RFOREST_EMB, MAE_MLP_EMB, MAE_LINEAR_EMB]}\n
        """)
        file.write("\n-----------------------------\n\n")
        file.write(f"""
    R2 GB: {R2_GB_EMB}\n
    R2 RFOREST: {R2_RFOREST_EMB}\n
    R2 MLP: {R2_MLP_EMB}\n
    R2 LINEAR: {R2_LINEAR_EMB}\n
    ALL R2: {[R2_GB_EMB, R2_RFOREST_EMB, R2_MLP_EMB, R2_LINEAR_EMB]}\n
        """)
        file.close()
