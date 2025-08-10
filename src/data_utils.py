from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def mice(df):
    df_columns = df.columns
    imputer = IterativeImputer(random_state=100, max_iter=100, keep_empty_features=True)
    imputer.fit(df[df_columns])
    imputed_value = imputer.transform(df[df_columns])
    mice_df = df.copy()
    mice_df.loc[:, df_columns] = imputed_value
    return mice_df
