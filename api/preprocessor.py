from src.data_utils import mice
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# GLOBAL VARIABLES
LABEL = ['pm25']
STAT_FEATS = ['lat', 'lon', 'station', 'pop', 'road_den_1km', 'prim_road_len_1km', 'near_dist', 'bareland', 'builtup', 'cropland', 'grassland', 'treecover', 'water', 'ndvi']
DYN_FEATS = ['aod', 'hpbl', 'wspd', 'rh', 'tmp']

# =========================================================== INTERNAL USE ===========================================================

# Fill the static features per station
# Static features does not change over station
def fill_static_features(df, stat_feat):
    all_filled_dfs = []
    for station in df["station"].unique():
        df_current_station = df[df["station"] == station]
        df_current_station[stat_feat] = df_current_station[stat_feat][0]
        all_filled_dfs.append(df_current_station)
    return pd.concat(all_filled_dfs, axis=0)

# Fill the dynamic features per station
# Dynamic features change over staiton, so we use "mice"
def fill_dynamic_features(df):
    all_filled_dfs = []
    for station in df["station"].unique():
        df_current_station = df[df["station"] == station]
        df_current_station_imputed = mice(df_current_station)
        all_filled_dfs.append(df_current_station_imputed)
    return pd.concat(all_filled_dfs, axis=0)
    
# =========================================================== PUBLIC USE ===========================================================

def fill_missing(df):
    null_stat = df.isnull().sum()
    df_result = df
    for feat in STAT_FEATS:
        if null_stat.loc[feat] != 0:
            df_result = fill_static_features(df_result, stat_feat)
    fill_dynamic = False
    for feat in DYN_FEATS:
        if null_stat.loc[feat] != 0:
            fill_dynamic = True
    if fill_dynamic:
        df_result = fill_dynamic_features(df_result)
    return df_result

def normalize_data(df):
    # Get the features
    features = STAT_FEATS + DYN_FEATS
    features.remove("station")
    # Get X and y
    X = df.loc[:, features]
    y = df.loc[:, LABEL]
    # Scale features
    features_scaler = MinMaxScaler()
    X_scaled = features_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    # Scale label
    label_scaler = MinMaxScaler()
    y_scaled = label_scaler.fit_transform(y)
    y_scaled = pd.DataFrame(y_scaled, columns=y.columns)
    # Return values
    return X_scaled, y_scaled, label_scaler
