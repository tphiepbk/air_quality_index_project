# Author: tphiepbk

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

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
