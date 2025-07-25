# Author: tphiepbk

import pandas as pd
import numpy as np

# ==========================================================================================

def splitTrainTestTimeSeries(X, y, test_percentage=0.2):
    total_size = X.shape[0]
    train_test_split_indicator_index = round(total_size * (1 - test_percentage))
    
    X_train = X[:train_test_split_indicator_index]
    y_train = y[:train_test_split_indicator_index]
    X_test = X[train_test_split_indicator_index:]
    y_test = y[train_test_split_indicator_index:]
    return X_train, X_test, y_train, y_test

# ==========================================================================================

'''
Reframe the dataset to past-future form
Length after reframing should be: total_size - n_past - n_future + 1
When keep_label_only is True, only the label column will be kept in the future sequence
'''
def reframePastFuture(df, n_past=1, n_future=1, keep_label_only=False):
    assert isinstance(df, pd.DataFrame), "df should be a DataFrame"

    total_len = len(df)
    ret_X, ret_y = [], []

    for window_start in range(total_len):
        past_end = window_start + n_past
        future_end = past_end + n_future
        
        if future_end > total_len:
              break
            
        ret_X.append(df.iloc[window_start:past_end, :])
        if keep_label_only:
              ret_y.append(df.iloc[past_end:future_end, 0])
        else:
              ret_y.append(df.iloc[past_end:future_end, :])

    if keep_label_only:
        ret_y = np.expand_dims(ret_y, axis=-1)
    
    return np.array(ret_X), np.array(ret_y)

# ==========================================================================================

def prepareReducedData(X_scaled, time_indices, station_column=None):
    # Deep copy
    df_aod_reduced = X_scaled.copy(deep=True)

    # Rename columns
    rename_dict = {column: f"aod_feature_{column+1}" for column in df_aod_reduced.columns}
    df_aod_reduced.rename(rename_dict, axis=1, inplace=True)
    
    # Set station column
    if station_column is not None:
        df_aod_reduced["station"] = station_column.values

    # Set time indices
    df_aod_reduced.set_index(time_indices, inplace=True)
    return df_aod_reduced
