# Author: tphiepbk

from keras import Input, Model
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError

from .base import Seq2SeqReductionModel

# Sequence-to-sequence reduction model using GRU
class CNNLSTMSeq2SeqReduction(Seq2SeqReductionModel):
    # Constructor
    def __init__(self, X_scaled, test_percentage=0.2, latent_dim=8, epochs=10, batch_size=10, n_past=0, n_future=0, verbose=0, model_name=None):
        super.__init__(X_scaled, test_percentage, latent_dim, epochs, batch_size, n_past, n_future, verbose, model_name)
        self._class_name = "CNNLSTMSeq2SeqReduction"

    # Define the model
    def _define_model(self):
        print(f"{self._class_name}._define_model(): is called") if self._verbose else None
        # Encoder layers
        encoder_inputs = Input(shape=(self._n_past, self._n_features))
        encoder_cnn_1 = Conv1D(filters=24, kernel_size=3, activation="relu")(encoder_inputs)
        encoder_cnn_2 = Conv1D(filters=12, kernel_size=3, activation="relu")(encoder_cnn_1)
        encoder_max_pooling = MaxPooling1D(pool_size=2)(encoder_cnn_2)
        encoder_flatten = Flatten()(encoder_max_pooling)
        encoder_repeat_vector = RepeatVector(self._n_future)(encoder_flatten)
        encoder_outputs, state_h, state_c = LSTM(50, return_state=True, activation="relu")(encoder_repeat_vector)
        encoder_dense = Dense(self._latent_dim)(encoder_outputs)
        # Repeat layer
        decoder_repeat_vector = RepeatVector(self._n_future)(encoder_dense)
        # Decoder layers
        decoder_lstm_1 = LSTM(50, return_sequences=True, activation="relu")(decoder_repeat_vector, initial_state=[state_h, state_c])
        decoder_dense_1 = TimeDistributed(Dense(24, activation="relu"))(decoder_lstm_1)
        decoder_outputs = TimeDistributed(Dense(self._n_features))(decoder_dense_1)
        # Compile the model
        cnn_lstm_seq2seq = Model(encoder_inputs, decoder_outputs)
        cnn_lstm_seq2seq.compile(optimizer=Adam(learning_rate=0.001), loss=MeanAbsoluteError())
        cnn_lstm_seq2seq
        return cnn_lstm_seq2seq
    
    # Define the Encoder model
    def _define_encoder_model(self):
        print(f"{self._class_name}._define_encoder_model(): is called") if self._verbose else None
        # Encoder only
        encoder_inputs = Input(shape=(self._n_past, self._n_features))
        encoder_cnn_1 = self._model.layers[1](encoder_inputs)
        encoder_cnn_2 = self._model.layers[2](encoder_cnn_1)
        encoder_max_pooling = self._model.layers[3](encoder_cnn_2)
        encoder_flatten = self._model.layers[4](encoder_max_pooling)
        encoder_repeat_vector = self._model.layers[5](encoder_flatten)
        encoder_outputs, _, _ = self._model.layers[6](encoder_repeat_vector)
        encoder_dense = self._model.layers[7](encoder_outputs)
        # Compile the model
        encoder_cnn_lstm_s2s = Model(encoder_inputs, encoder_dense)
        encoder_cnn_lstm_s2s.compile(optimizer=Adam(), loss=MeanAbsoluteError())
        encoder_cnn_lstm_s2s.name = self._model.name + "_encoder"
        return encoder_cnn_lstm_s2s
