# Author: tphiepbk

from keras import Input, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError

from src.reduction_model.base import Seq2SeqReductionModel

# Sequence-to-sequence reduction model using LSTM
class LSTMSeq2SeqReduction(Seq2SeqReductionModel):
    # Class attributes
    class_name = "LSTMSeq2SeqReduction"
    
    # Constructor
    def __init__(self, scaled_data, val_percentage=0.2, test_percentage=0.2, latent_dim=8, epochs=10, batch_size=10, n_past=0, n_future=0, verbose=0, model_name=None):
        super().__init__(scaled_data, val_percentage, test_percentage, latent_dim, epochs, batch_size, n_past, n_future, verbose, model_name, class_name=self.__class__.class_name)

    # Define the model
    def _define_model(self):
        print(f"{self.__class__.class_name}._define_model(): is called") if self._verbose else None
        # Encoder layers
        encoder_inputs = Input(shape=(self._n_past, self._n_features))
        encoder_lstm_1 = LSTM(100, return_sequences=True, activation="relu")(encoder_inputs)
        encoder_outputs, state_h, state_c = LSTM(50, return_state=True, activation="relu")(encoder_lstm_1)
        encoder_dense = Dense(self._latent_dim)(encoder_outputs)
        # Repeat layer
        decoder_repeat_vector = RepeatVector(self._n_future)(encoder_dense)
        # Decoder layers
        decoder_lstm_1 = LSTM(50, return_sequences=True, activation="relu")(decoder_repeat_vector, initial_state=[state_h, state_c])
        decoder_lstm_2 = LSTM(100, return_sequences=True, activation="relu")(decoder_lstm_1)
        decoder_outputs = TimeDistributed(Dense(self._n_features))(decoder_lstm_2)
        # Compile the model
        lstm_seq2seq = Model(encoder_inputs, decoder_outputs)
        lstm_seq2seq.compile(optimizer=Adam(learning_rate=0.001), loss=MeanAbsoluteError())
        return lstm_seq2seq

    # Define the Encoder model
    def _define_encoder_model(self):
        print(f"{self.__class__.class_name}._define_encoder_model(): is called") if self._verbose else None
        # Encoder only
        encoder_inputs = Input(shape=(self._n_past, self._n_features))
        encoder_lstm_1 = self._model.layers[1](encoder_inputs)
        encoder_outputs, _, _ = self._model.layers[2](encoder_lstm_1)
        encoder_dense = self._model.layers[3](encoder_outputs)
        # Compile the model
        encoder_lstm_s2s = Model(encoder_inputs, encoder_dense)
        encoder_lstm_s2s.compile(optimizer=Adam(learning_rate=0.001), loss=MeanAbsoluteError())
        encoder_lstm_s2s.name = self._model.name + "_encoder"
        return encoder_lstm_s2s
