from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

def mice(df, method=None):
    if method is not None:
        assert method in ["random_forest", "extra_trees"]
    df_columns = df.columns
    if method == "random_forest":
        estimator = RandomForestRegressor()
        max_iter = 10
    elif method == "extra_trees":
        estimator = ExtraTreesRegressor()
        max_iter = 10
    else:
        estimator = BayesianRidge()
        max_iter = 100
    imputer = IterativeImputer(estimator=estimator, random_state=100, max_iter=max_iter, keep_empty_features=True)
    imputer.fit(df[df_columns])
    imputed_value = imputer.transform(df[df_columns])
    mice_df = df.copy()
    mice_df.loc[:, df_columns] = imputed_value
    return mice_df
