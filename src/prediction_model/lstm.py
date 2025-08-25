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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras import Input, Model, Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from keras.utils import plot_model
from keras.saving import load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError
from keras.losses import MeanAbsoluteError, MeanSquaredError
import keras.backend as K

from src.time_series_utils import splitTrainValidationTestTimeSeries, reframePastFuture, padPastFuture
from src.config_reader import ConfigurationReader
from src.plot import plot_learning_curves

conf = ConfigurationReader("/le_thanh_van_118/workspace/hiep_workspace/air_quality_index_project/model_params.json").data

'''
Receive the encoded features and labels as inputs
This implementation could be applied for any n_past and n_future
'''
class LSTMPrediction(object):
    # Class attribute
    class_name = 'LSTMPrediction'

    # Class method
    @classmethod
    def get_class_name(cls):
        return cls.class_name
    
    def __init__(self, X_scaled: pd.DataFrame, y_scaled: pd.DataFrame, label_scaler, val_percentage=0.2, test_percentage=0.2, epochs=10, batch_size=10, n_past=0, n_future=0, verbose=0, model_name=None):
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
        dropout_2 = Dropout(0.2)(decoder_lstm)
        decoder_dense_1 = TimeDistributed(Dense(32, activation="relu"))(dropout_2)
        decoder_dense_2 = TimeDistributed(Dense(self._y_scaled_reframed.shape[-1]))(decoder_dense_1)
        model = Model(encoder_input, decoder_dense_2)
        # Compile model
        model.compile(loss=MeanAbsoluteError(), optimizer=Adam(learning_rate=0.001))
        return model

    # Get model information
    def get_model_info(self):
        print(self._model.summary())
        plot_model(self._model, to_file=f'{conf["workspace"]["model_info_dir"]}/{self._model.name}.png', show_shapes=True, dpi=100)

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
        # Define model checkpoint
        checkpoint = ModelCheckpoint(filepath=f'{conf["workspace"]["model_checkpoints_dir"]}/{self._model.name}.keras', save_best_only=True)
        # Fit model
        history = self._model.fit(X_train_reframed, y_train_reframed,
                epochs=self._epochs,
                batch_size=self._batch_size,
                validation_data=(X_val_reframed, y_val_reframed),
                verbose=self._verbose,
                callbacks=[checkpoint],
                shuffle=False)
        # Plot the learning curves
        plot_learning_curves(history)
        # Predict test data
        y_predicted = self._model.predict(X_test_reframed, verbose=self._verbose)
        return y_predicted, y_test_reframed

    # Evaluate the model
    def _evaluate_model(self, y_pred, y_test):
        # The scaling inverted of the y_test and y_pred of all days (n_future)
        all_days_inv_y_test, all_days_inv_y_pred = [], []
        
        # The MAE, MSE, R2 of all days
        all_days_mae = []
        all_days_mse = []
        all_days_r2 = []
        
        # In case the y_pred has this shape: (n_predict_samples, n_future),
        # expand the last dimension => (n_predict_samples, n_future, n_label)
        # n_label should be 1
        if len(y_pred.shape) < 3:
            y_pred = np.expand_dims(y_pred, axis=-1)
        
        print(f"y_pred.shape = {y_pred.shape}\ny_test.shape = {y_test.shape}") if self._verbose else None
        
        for day in range(self._n_future):
            current_day_y_pred = y_pred[:,day]
            current_day_inv_y_pred = self._label_scaler.inverse_transform(current_day_y_pred)
            all_days_inv_y_pred.append(current_day_inv_y_pred)
            
            current_day_y_test = y_test[:,day]
            current_day_inv_y_test = self._label_scaler.inverse_transform(current_day_y_test)
            all_days_inv_y_test.append(current_day_inv_y_test)
    
            mae = mean_absolute_error(current_day_inv_y_pred, current_day_inv_y_test)
            all_days_mae.append(mae)
    
            mse = mean_squared_error(current_day_inv_y_pred, current_day_inv_y_test)
            all_days_mse.append(mse)
    
            r2 = r2_score(current_day_inv_y_pred, current_day_inv_y_test)
            all_days_r2.append(r2)
        
            print(f"Day {day+1} - mae = {mae}, mse = {mse}, r2 = {r2}") if self._verbose else None
        
        # Convert to NumPy array
        # Output shape:
        #   - all_days_mae: (n_future,)
        #   - all_days_inv_y_pred: (n_future, n_predict_samples, n_label)
        #   - all_days_inv_y_test: (n_future, n_test_samples, n_label)
        all_days_mae = np.array(all_days_mae)
        all_days_mse = np.array(all_days_mse)
        all_days_r2 = np.array(all_days_r2)
        all_days_inv_y_pred = np.array(all_days_inv_y_pred)
        all_days_inv_y_test = np.array(all_days_inv_y_test)
        
        # Calculate the average MAE, MSE, R2 of all days
        avg_mae = np.average(all_days_mae)
        print(f"avg_mae = {avg_mae}") if self._verbose else None
        avg_mse = np.average(all_days_mse)
        print(f"avg_mse = {avg_mse}") if self._verbose else None
        avg_r2 = np.average(all_days_r2)
        print(f"avg_r2 = {avg_r2}") if self._verbose else None
        
        # Return the avarage MAE of all days and the MAE of each day also
        # Return the inverted y_test and y_pred of each day also
        return all_days_inv_y_pred, all_days_inv_y_test, all_days_mae, avg_mae, all_days_mse, avg_mse, all_days_r2, avg_r2
        
    # Main execution method
    def execute(self):
        # Set logging to ERROR only
        tf.get_logger().setLevel(logging.ERROR)
        print(f"{self.__class__.class_name}.execute(): is called") if self._verbose else None
        # Train model
        y_predicted, y_test_reframed = self._train_model()
        # Set logging to INFO only
        tf.get_logger().setLevel(logging.INFO)
        # Return the evaluation model        
        return self._evaluate_model(y_predicted, y_test_reframed)
    
# ==========================================================================================

'''
Receive the encoded features and labels as inputs
This implementation could be applied for any n_past and n_future
Do not split to train test, just predict the encoded data
Return the predicted label values
'''
def predictLSTMNoSplit(X, y, n_past=1, n_future=1, epochs=10, batch_size=64, model_name="lstm", verbose=0):
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
    
    # Define model checkpoint
    checkpoint = ModelCheckpoint(filepath=f'{conf["workspace"]["model_checkpoints_dir"]}/{model_name}.keras', save_best_only=True)
    
    # Define LSTM model
    # Input shape: (n_past, n_features)
    # Output: (n_future, n_label)
    n_features, n_label = X_reframed.shape[-1], y_reframed.shape[-1]
    
    # Define layers
    model = Sequential()
    encoder_input = Input(shape=(n_past, n_features))
    encoder_lstm, state_h, state_c = LSTM(64, activation="relu", return_state=True)(encoder_input)
    decoder_input = RepeatVector(n_future)(encoder_lstm)
    decoder_lstm = LSTM(64, activation="relu", return_sequences=True)(decoder_input, initial_state = [state_h, state_c])
    dropout_2 = Dropout(0.2)(decoder_lstm)
    decoder_dense_1 = TimeDistributed(Dense(32, activation="relu"))(dropout_2)
    decoder_dense_2 = TimeDistributed(Dense(n_label))(decoder_dense_1)
    model = Model(encoder_input, decoder_dense_2)
    
    # Compile model
    model.compile(loss=MeanAbsoluteError(), optimizer=Adam(learning_rate=0.001))
    model.name = model_name
    print(model.summary()) if verbose else None
    plot_model(model, to_file=f'{conf["workspace"]["model_info_dir"]}/{model_name}.png', show_shapes=True, dpi=100)
    
    # Fit model
    if verbose:
        print(f"Training model {model_name}...")
    model.fit(X_reframed, y_reframed,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[checkpoint],
            shuffle=False)
    print("=" * 100) if verbose else None
    
    # Prediction
    # The shape of y_pred should be [n_predicted_samples, n_future, n_label]
    print(f"Predicting using model {model_name}...") if verbose else None
    y_pred = model.predict(X_reframed, verbose=verbose)
    print("=" * 100) if verbose else None
    
    return y_pred