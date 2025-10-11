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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
from keras import Input, Model, Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from tqdm.keras import TqdmCallback
from keras.models import load_model
from scipy.stats import pearsonr

from src.time_series_utils import splitTrainValidationTestTimeSeries, reframePastFuture, padPastFuture
from src.config_reader import ConfigurationReader
from src.plot import plot_learning_curves


conf = ConfigurationReader("/le_thanh_van_118/workspace/hiep_workspace/air_quality_index_project/model_params.json").data

# Calculate MNBE
def mean_normalized_bias_error(y_pred, y_actual):
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)
    return np.mean((y_pred - y_actual) / np.mean(y_actual)) * 100

'''
Receive the encoded features and labels as inputs
This implementation could be applied for any n_past and n_future
'''
class LSTMPrediction(object):
    # Class attribute
    class_name = 'LSTMPrediction'
    supported_metrics = {"mae": mean_absolute_error,
                         "mse": mean_squared_error,
                         "rmse": root_mean_squared_error,
                         "r2": r2_score,
                         "mape": mean_absolute_percentage_error,
                         "mnbe": mean_normalized_bias_error,
                         "r_coeff": pearsonr,
                         "p_value": pearsonr}

    # Class method
    @classmethod
    def get_class_name(cls):
        return cls.class_name

    @classmethod
    def get_supported_metrics(cls):
        return cls.supported_metrics
    
    def __init__(self, X_scaled: pd.DataFrame, y_scaled: pd.DataFrame,
                 label_scaler,
                 val_percentage=0.2, test_percentage=0.2,
                 n_past=0, n_future=0,
                 epochs=10, batch_size=10,
                 verbose=0,
                 model_name=None):
        # Hyper parameters
        self._verbose = verbose
        self._label_scaler = label_scaler
        self._test_percentage = test_percentage
        self._val_percentage = val_percentage
        self._epochs = epochs
        self._batch_size = batch_size
        self._n_past = n_past
        self._n_future = n_future
        # Data
        self._X_scaled = X_scaled
        self._y_scaled = y_scaled
        self._X_scaled_reframed = None
        self._y_scaled_reframed = None
        self._prepare_data()
        # Models
        self._model = self._define_model()
        if model_name:
              self._model.name = model_name
    
    def _prepare_data(self):
        print(f"{self.__class__.class_name}._prepare_data(): is called") if self._verbose else None
        # Combine self._X_scaled and self._y_scaled, self._y_scaled should be the last column
        combined_df = pd.concat((self._X_scaled, self._y_scaled), axis=1)
        # Reframe the dataset to past-future form
        # Output shape:
        #   - n_samples_reframed = n_samples - n_past - n_future + 1
        #   - n_features_label = n_features + n_label
        #   - self._X_scaled_reframed: [n_samples_reframed, n_past, n_features_label]
        #   - self._y_scaled_reframed: [n_samples_reframed, n_future, n_label]
        self._X_scaled_reframed, self._y_scaled_reframed = reframePastFuture(combined_df, self._n_past, self._n_future, keep_label_only=True)

    # Define the LSTM model
    def _define_model(self):
        # Define layers
        model = Sequential()
        encoder_input = Input(shape=(self._n_past, self._X_scaled_reframed.shape[-1]))
        encoder_lstm, state_h, state_c = LSTM(64, activation="relu", return_state=True)(encoder_input)
        decoder_input = RepeatVector(self._n_future)(encoder_lstm)
        decoder_lstm = LSTM(64, activation="relu", return_sequences=True)(decoder_input, initial_state = [state_h, state_c])
        dropout = Dropout(0.2)(decoder_lstm)
        decoder_dense = TimeDistributed(Dense(self._y_scaled_reframed.shape[-1]))(dropout)
        model = Model(encoder_input, decoder_dense)
        # Compile model
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))
        return model

    # Train the model
    def _train_model(self):
        print(f"{self.__class__.class_name}._train_model(): is called") if self._verbose else None
        
        # Split self._X_scaled_reframed and self._y_scaled_reframed into train set, validation set and test set
        # Output shape:
        #   - X_train_reframed: [n_train_samples, n_past, n_features]
        #   - y_train_reframed: [n_train_samples, n_future, n_label]
        #   - X_val_reframed: [n_val_samples, n_past, n_features]
        #   - y_val_reframed: [n_val_samples, n_future, n_label]
        #   - X_test_reframed: [n_test_samples, n_past, n_features]
        #   - y_test_reframed: [n_test_samples, n_future, n_label]
        X_train_reframed, X_val_reframed, X_test_reframed, y_train_reframed, y_val_reframed, y_test_reframed = splitTrainValidationTestTimeSeries(self._X_scaled_reframed, self._y_scaled_reframed, test_percentage=self._test_percentage, val_percentage=self._val_percentage)
        # Fit model
        history = self._model.fit(X_train_reframed, y_train_reframed,
                epochs=self._epochs,
                batch_size=self._batch_size,
                validation_data=(X_val_reframed, y_val_reframed),
                verbose=self._verbose,
                callbacks = [
                    EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True),
                    TqdmCallback(verbose=1)
                ],
                shuffle=False)
        # Plot the learning curves
        plot_learning_curves(history)
        # Predict test data
        y_predicted = self._model.predict(X_test_reframed, verbose=self._verbose)
        return y_predicted, y_test_reframed

    # Evaluate the model
    def _evaluate_model(self, y_pred, y_test):
        # Shape of y_pred and y_test should be (n_predict_samples, n_future, n_label),
        # where n_label should be 1
        print(f"y_pred.shape = {y_pred.shape}\ny_test.shape = {y_test.shape}") if self._verbose else None
        
        # A dictionary containing all metrics
        # For each metric type, the first element is a list of all days' metric,
        # while the second element is the average value of all days, but we will update it later
        metrics = {metric: [] for metric in LSTMPrediction.get_supported_metrics().keys()}
        print(f"Initialize metrics = {metrics}") if self._verbose else None

        # Inverse transform
        inv_y_pred = self._label_scaler.inverse_transform(np.squeeze(y_pred, axis=-1))
        inv_y_test = self._label_scaler.inverse_transform(np.squeeze(y_test, axis=-1))
        print(f"inv_y_pred.shape = {inv_y_pred.shape}, inv_y_test.shape = {inv_y_test.shape}") if self._verbose else None

        # Calculate metrics for all days
        for day in range(self._n_future):
            current_day_inv_y_pred = inv_y_pred[:, day]
            current_day_inv_y_test = inv_y_test[:, day]

            print(f"Day {day+1}:") if self._verbose else None
            for metric_type, calculator in LSTMPrediction.get_supported_metrics().items():
                calculated_metric_value = calculator(current_day_inv_y_pred, current_day_inv_y_test)
                
                if metric_type == "r_coeff":
                    calculated_metric_value = calculated_metric_value[0]
                elif metric_type == "p_value":
                    calculated_metric_value = calculated_metric_value[1]
                    
                metrics[metric_type].append(calculated_metric_value)
                print(f"\t{metric_type} = {calculated_metric_value}") if self._verbose else None

        # Calculate average metrics and update the metrics dictionary
        for metric_type, all_days_metric in metrics.items():
            avg_metric_value = np.average(np.array(all_days_metric))
            metrics[metric_type] = (all_days_metric, avg_metric_value)
            print(f"avg_{metric_type} = {avg_metric_value}") if self._verbose else None
        
        # Return the avarage metrics of all days and the metrics of each day also
        # Return the inverted y_test and y_pred of each day also
        # return all_days_inv_y_pred, all_days_inv_y_test, metrics
        return inv_y_pred, inv_y_test, metrics

    # Get model information
    def dump(self, saved_model_plot_dir):
        print(self._model.summary())
        
    # Main execution method
    def execute(self, saved_model_weight_dir):
        # Set logging to ERROR only
        tf.get_logger().setLevel(logging.ERROR)
        print(f"{self.__class__.class_name}.execute(): is called") if self._verbose else None
        # Train model
        y_predicted, y_test_reframed = self._train_model()
        # Set logging to INFO only
        tf.get_logger().setLevel(logging.INFO)
        # Save model
        model_path = f"{saved_model_weight_dir}/{self._model.name}.keras"
        self._model.save(model_path, include_optimizer=True)
        # Return the evaluation model        
        return *self._evaluate_model(y_predicted, y_test_reframed), model_path

# ==========================================================================================

'''
Inference data using saved LSTM model
'''
def inferenceLSTM(X, y, n_past=1, n_future=1, saved_model_weight_dir=".", verbose=0):
    # Set logging to ERROR only
    tf.get_logger().setLevel(logging.ERROR)

    # Padding
    X = padPastFuture(X, n_past, n_future)
    y = padPastFuture(y, n_past, n_future)
    
    # Convert to numpy array
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()
    
    # Combine X and y, y should be the last column
    combined_X_y = np.concatenate((X, y), axis=1)
    combined_df = pd.DataFrame(combined_X_y)
    
    # Reframe the dataset to past-future form
    # Output shape:
    #   - n_samples_reframed = n_samples - n_past - n_future + 1
    #   - n_features_label = n_features + n_label
    #   - X_reframed: [n_samples_reframed, n_past, n_features_label]
    #   - y_reframed: [n_samples_reframed, n_future, n_label]
    X_reframed, y_reframed = reframePastFuture(combined_df, n_past, n_future, keep_label_only=True)

    # Load model and predict
    saved_model = load_model(saved_model_weight_dir)
    y_pred = saved_model.predict(X_reframed, verbose=verbose)
    print("=" * 100) if verbose else None
    return y_pred
