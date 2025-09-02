# Author: tphiepbk

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# ==========================================================================================

def padPastFuture(data: pd.DataFrame, n_past=1, n_future=1):
    padded_before = pd.DataFrame([data.iloc[0]] * n_past)
    padded_after = pd.DataFrame([data.iloc[-1]] * (n_future - 1))
    return pd.concat([padded_before, data, padded_after], axis=0)

# ==========================================================================================

def splitTrainValidationTestTimeSeries(X, y, test_percentage=0.2, val_percentage=0.2):
    total_size = X.shape[0]
    train_test_split_indicator_index = round(total_size * (1 - test_percentage))
    
    X_train_val = X[:train_test_split_indicator_index]
    y_train_val = y[:train_test_split_indicator_index]
    X_test = X[train_test_split_indicator_index:]
    y_test = y[train_test_split_indicator_index:]

    train_val_size = len(X_train_val)
    train_val_split_indicator_index = round(train_val_size * (1 - val_percentage))
    
    X_train = X[:train_val_split_indicator_index]
    y_train = y[:train_val_split_indicator_index]
    X_val = X[train_val_split_indicator_index:]
    y_val = y[train_val_split_indicator_index:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

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
            ret_y.append(df.iloc[past_end:future_end, -1])
        else:
            ret_y.append(df.iloc[past_end:future_end, :])

    if keep_label_only:
        ret_y = np.expand_dims(ret_y, axis=-1)
    
    return np.array(ret_X), np.array(ret_y)

# ==========================================================================================
# Check the stationarity of a timeseries
# Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

def check_stationarity(series):
    result = adfuller(series.values)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

