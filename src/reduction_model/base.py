# Author: tphiepbk

# System
import os
import glob
import shutil
import copy
import re
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

conf = ConfigurationReader("/le_thanh_van_118/workspace/hiep_workspace/air_quality_index_project/model_params.json").data

# ==========================================================================================

# Base class for every other reduction model
class Seq2SeqReductionModel(object):
    def __init__(self, X_scaled, test_percentage=0.2, latent_dim=8, epochs=10, batch_size=10, n_past=0, n_future=0, verbose=0, model_name=None, class_name=None):
        # Hyper parameters
        self._verbose = verbose
        self._test_percentage = test_percentage
        self._latent_dim = latent_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._n_past = n_past
        self._n_future = n_future
        # Metadata of the object
        if class_name:
            self._class_name = class_name
        else:
            self._class_name = "Seq2SeqReductionModel"
        # Data
        self._X_scaled = X_scaled
        self._X_scaled_reframed = None
        self._y_scaled_reframed = None
        self._n_features = self._X_scaled.shape[1]
        self._prepare_data()
        # Models
        self._model = self._define_model()
        if model_name:
              self._model.name = model_name
        self._encoder_model = None
    
    def _prepare_data(self):
        print(f"{self._class_name}._prepare_data(): is called") if self._verbose else None
        # Padding data
        padded_before = pd.DataFrame([self._X_scaled.iloc[0]] * self._n_past)
        padded_after = pd.DataFrame([self._X_scaled.iloc[-1]] * (self._n_future - 1))
        X_scaled_padded = pd.concat([padded_before, self._X_scaled, padded_after], axis=0, ignore_index=True)
        # Reframe data
        self._X_scaled_reframed, self._y_scaled_reframed = reframePastFuture(X_scaled_padded, self._n_past, self._n_future)

    # Get model information
    def get_model_info(self):
        print(self._model.summary())
        plot_model(self._model, to_file=f'{conf["workspace"]["model_info_dir"]}/{self._model.name}.png', show_shapes=True, dpi=100)

    # Get encoder model information
    def get_encoder_model_info(self):
        print(self._encoder_model.summary())
        plot_model(self._encoder_model, to_file=f'{conf["workspace"]["model_info_dir"]}/{self._encoder_model.name}.png', show_shapes=True, dpi=100)

    # Main execution method
    def execute(self):
        # Set logging to ERROR only
        tf.get_logger().setLevel(logging.ERROR)
        print(f"{self._class_name}.execute(): is called") if self._verbose else None
        mae = self._train_model()
        print(f"{self._class_name}.execute(): mae = {mae}") if self._verbose else None
        self._encoder_model = self._define_encoder_model()
        # Set logging to INFO only
        tf.get_logger().setLevel(logging.INFO)
        return self._encode_data(), self._save_encoder_model()

    # Train and evaluate the model
    def _train_model(self):
        print(f"{self._class_name}._train_model(): is called") if self._verbose else None
        X_train, X_test, y_train, y_test = splitTrainTestTimeSeries(self._X_scaled_reframed, self._y_scaled_reframed, test_percentage=self._test_percentage)
        self._model.fit(X_train, y_train,
                        epochs=self._epochs,
                        batch_size=self._batch_size,
                        validation_data=(X_test, y_test),
                        shuffle=False,
                        verbose=self._verbose)
        y_predicted= self._model.predict(X_test, verbose=self._verbose)
        mae = self._model.evaluate(y_predicted, y_test, verbose=self._verbose)
        return mae

    # Get save the encoder model and return the path
    def _save_encoder_model(self):
        model_path = f'{conf["workspace"]["model_info_dir"]}/{self._encoder_model.name}.keras'
        self._encoder_model.save(model_path, include_optimizer=True)
        return model_path
        
    # Reduce dimension with trained Encoder
    def _encode_data(self):
        print(f"{self._class_name}._encode_data(): is called") if self._verbose else None
        return pd.DataFrame(self._encoder_model.predict(self._X_scaled_reframed, verbose=self._verbose))

    # Define the Encoder model
    # Override in derived classes
    def _define_encoder_model(self):
        pass
        
    # Define the model
    # Override in derived classes
    def _define_model(self):
        pass
