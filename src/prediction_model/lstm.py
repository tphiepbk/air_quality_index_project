# Author: tphiepbk

# System
import os
import glob
import shutil
import copy
import re
from datetime import datetime
import logging

# Data
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Data processing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Model
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from keras import Input, Model, Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from keras.utils import plot_model
from keras.saving import load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError
from keras.losses import MeanAbsoluteError, MeanSquaredError
import keras.backend as K

from src.time_series_utils import splitTrainTestTimeSeries, reframePastFuture
from src.config_reader import ConfigurationReader

global_conf = ConfigurationReader("/le_thanh_van_118/workspace/hiep_workspace/model_params.json")

# ==========================================================================================

'''
Receive the encoded features and labels as inputs
This implementation could be applied for any n_past and n_future
'''
def predictLSTM(X, y, n_past=1, n_future=1, epochs=10, batch_size=64, model_name="lstm", verbose=0):
  # Set logging to ERROR only
  tf.get_logger().setLevel(logging.ERROR)
    
  # Convert to numpy array
  if isinstance(X, pd.DataFrame):
    X = X.to_numpy()
  if isinstance(y, pd.DataFrame):
    y = y.to_numpy()

  # Combine X and y, y should be the first column
  combined_y_X = np.concatenate((y, X), axis=1)
  combined_df = pd.DataFrame(combined_y_X)

  # Reframe the dataset to past-future form
  # Output shape:
  #   - n_samples_reframed = n_samples - n_past - n_future + 1
  #   - n_features_label = n_features + n_label
  #   - X_reframed: [n_samples_reframed, n_past, n_features_label]
  #   - y_reframed: [n_samples_reframed, n_future, n_label]
  X_reframed, y_reframed = reframePastFuture(combined_df, n_past, n_future, keep_label_only=True)

  # Split df_X and df_y into train set and test set
  # Output shape:
  #   - X_train_reframed: [n_train_samples, n_past, n_features]
  #   - y_train_reframed: [n_train_samples, n_future, n_label]
  #   - X_test_reframed: [n_test_samples, n_past, n_features]
  #   - y_test_reframed: [n_test_samples, n_future, n_label]
  X_train_reframed, X_test_reframed, y_train_reframed, y_test_reframed = splitTrainTestTimeSeries(X_reframed, y_reframed, test_percentage=0.2)

  # Define model checkpoint
  checkpoint = ModelCheckpoint(filepath=f'{global_conf.general["model_checkpoints_dir"]}/{model_name}.keras', save_best_only=True)

  # Define LSTM model
  # Input shape: (n_past, n_features)
  # Output: (n_future, n_label)
  n_features, n_label = X_train_reframed.shape[-1], y_train_reframed.shape[-1]

  model = Sequential()
  encoder_input = Input(shape=(n_past, n_features))
  encoder_lstm, state_h, state_c = LSTM(200, activation="relu", return_state=True)(encoder_input)
  decoder_input = RepeatVector(n_future)(encoder_lstm)
  decoder_lstm = LSTM(200, activation="relu", return_sequences=True)(decoder_input, initial_state = [state_h, state_c])
  decoder_dense_1 = TimeDistributed(Dense(100, activation="relu"))(decoder_lstm)
  decoder_dense_2 = TimeDistributed(Dense(1))(decoder_dense_1)
  model = Model(encoder_input, decoder_dense_2)

  # Compile model
  model.compile(loss=MeanAbsoluteError(), optimizer=Adam(learning_rate=0.001))
  model.name = model_name
  print(model.summary()) if verbose else None
  plot_model(model, to_file=f'{global_conf.general["model_info_dir"]}/{model_name}.png', show_shapes=True, dpi=100)

  # Fit model
  if verbose:
    print(f"Training model {model_name}...")
  model.fit(X_train_reframed, y_train_reframed,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_reframed, y_test_reframed),
            verbose=verbose,
            callbacks=[checkpoint],
            shuffle=False)
  print("=" * 100) if verbose else None

  # Prediction
  # The shape of y_pred should be [n_predicted_samples, n_future, n_label]
  print(f"Predicting using model {model_name}...") if verbose else None
  y_pred = model.predict(X_test_reframed, verbose=verbose)
  print("=" * 100) if verbose else None

  return y_pred, y_test_reframed

# ==========================================================================================

# Evaluate LSTM model using MeanAbsoluteError
# This implementation could be applied for any n_past and n_future
def evaluateLSTM(y_pred, y_test, label_scaler, verbose=0):
  # The scaling inverted of the y_test and y_pred of all days (n_future)
  all_days_inv_y_test, all_days_inv_y_pred = [], []

  # The MAE of all days
  all_days_mae = []

  # In case the y_pred has this shape: (n_predict_samples, n_future),
  # expand the last dimension => (n_predict_samples, n_future, n_label)
  # n_label should be 1
  if len(y_pred.shape) < 3:
    y_pred = np.expand_dims(y_pred, axis=-1)

  print(f"y_pred.shape = {y_pred.shape}\ny_test.shape = {y_test.shape}") if verbose else None

  # n_future is the second shape of y_pred
  n_future = y_pred.shape[1]

  for day in range(n_future):
    current_day_y_pred = y_pred[:,day]
    current_day_inv_y_pred = label_scaler.inverse_transform(current_day_y_pred)
    all_days_inv_y_pred.append(current_day_inv_y_pred)

    current_day_y_test = y_test[:,day]
    current_day_inv_y_test = label_scaler.inverse_transform(current_day_y_test)
    all_days_inv_y_test.append(current_day_inv_y_test)

    mae = mean_absolute_error(current_day_inv_y_pred, current_day_inv_y_test)
    all_days_mae.append(mae)

    print(f"Day {day+1} - MAE = {mae}") if verbose else None

  # Convert to NumPy array
  # Output shape:
  #   - all_days_mae: (n_future,)
  #   - all_days_inv_y_pred: (n_future, n_predict_samples, n_label)
  #   - all_days_inv_y_test: (n_future, n_test_samples, n_label)
  all_days_mae = np.array(all_days_mae)
  all_days_inv_y_pred = np.array(all_days_inv_y_pred)
  all_days_inv_y_test = np.array(all_days_inv_y_test)

  # Calculate the average MAE of all days
  avg_mae = np.average(all_days_mae)
  print(f"avg_mae = {avg_mae}") if verbose else None

  # Set logging to INFO
  tf.get_logger().setLevel(logging.INFO)

  # Return the avarage MAE of all days and the MAE of each day also
  # Return the inverted y_test and y_pred of each day also
  return all_days_inv_y_pred, all_days_inv_y_test, all_days_mae, avg_mae
