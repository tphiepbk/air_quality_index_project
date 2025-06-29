# ==========================================================================================

class LSTMSeq2SeqReduction(object):
    def __init__(self, X_scaled, test_percentage=0.2, latent_dim=8, epochs=10, batch_size=10, n_past=0, n_future=0, verbose=0, model_name=None):
    # Hyper parameters
    self._verbose = verbose
    self._test_percentage = test_percentage
    self._latent_dim = latent_dim
    self._epochs = epochs
    self._batch_size = batch_size
    self._n_past = n_past
    self._n_future = n_future
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

    # Get model information
    def get_model_info(self):
        print(self._model.summary())
        plot_model(self._model, to_file=f"{model_info_dir}/{self._model.name}.png", show_shapes=True, dpi=100)

    # Get encoder model information
    def get_encoder_model_info(self):
        print(self._encoder_model.summary())
        plot_model(self._encoder_model, to_file=f"{model_info_dir}/{self._encoder_model.name}.png", show_shapes=True, dpi=100)

    # Main execution method
    def execute(self):
        # Set logging to ERROR only
        tf.get_logger().setLevel(logging.ERROR)
        print("LSTMSeq2SeqReduction.execute(): is called") if self._verbose else None
        mae = self._train_model()
        print(f"LSTMSeq2SeqReduction.execute(): mae = {mae}") if self._verbose else None
        self._encoder_model = self._define_encoder_model()
        # Set logging to INFO only
        tf.get_logger().setLevel(logging.INFO)
        return self._encode_data()
    
    def _prepare_data(self):
        print("LSTMSeq2SeqReduction._prepare_data(): is called") if self._verbose else None
        # Padding data
        padded_before = pd.DataFrame([self._X_scaled.iloc[0]] * self._n_past)
        padded_after = pd.DataFrame([self._X_scaled.iloc[-1]] * (self._n_future - 1))
        X_scaled_padded = pd.concat([padded_before, self._X_scaled, padded_after], axis=0, ignore_index=True)
        # Reframe data
        self._X_scaled_reframed, self._y_scaled_reframed = reframePastFuture(X_scaled_padded, self._n_past, self._n_future)

    # Define the model
    def _define_model(self):
        print("LSTMSeq2SeqReduction._define_model(): is called") if self._verbose else None
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

    # Train and evaluate the model
    def _train_model(self):
        print("LSTMSeq2SeqReduction._train_model(): is called") if self._verbose else None
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

    # Define the Encoder model
    def _define_encoder_model(self):
        print("LSTMSeq2SeqReduction._define_encoder_model(): is called") if self._verbose else None
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
    
    # Reduce dimension with trained Encoder
    def _encode_data(self):
        print("LSTMSeq2SeqReduction._encode_data(): is called") if self._verbose else None
        return pd.DataFrame(self._encoder_model.predict(self._X_scaled_reframed, verbose=self._verbose))
