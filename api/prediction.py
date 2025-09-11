from src.loop_model import generate_loopresults, choose_the_best

from src.prediction_model.lstm import LSTMPrediction, predictLSTMNoSplit
from src.reduction_model.lstm_s2s import LSTMSeq2SeqReduction
from src.reduction_model.gru_s2s import GRUSeq2SeqReduction
from src.reduction_model.cnnlstm_s2s import CNNLSTMSeq2SeqReduction

# Run the loop with LSTM Seq2Seq reduction model and LSTM prediction model
def predict_pm25_using_lstms2s_lstm(X_scaled, y_scaled,
                               label_scaler,
                               range_of_dimension,
                               reduction_n_past, reduction_n_future,
                               reduction_epochs, reduction_batch_size,
                               prediction_n_past, prediction_n_future,
                               prediction_epochs, prediction_batch_size):

    # Run the loop
    loopresults = generate_loopresults(X_scaled, y_scaled,
                                       label_scaler,
                                       range_of_dimension,
                                       LSTMSeq2SeqReduction,
                                       LSTMPrediction,
                                       reduction_n_past, reduction_n_future,
                                       reduction_epochs, reduction_batch_size,
                                       prediction_n_past, prediction_n_future,
                                       prediction_epochs, prediction_batch_size)

    # Get the best data
    best_metrics, best_num_of_components, best_encoded_data, best_evaluation_data, best_encoder_model_path = choose_the_best(loopresults,
                                                                                                                             metric_to_choose="mae",
                                                                                                                             prediction_n_future=prediction_n_future)

    # Predict using encoded data
    y_pred = predictLSTMNoSplit(best_encoded_data, y_scaled,
                                n_past=prediction_n_past, n_future=prediction_n_future,
                                epochs=prediction_epochs, batch_size=prediction_batch_size,
                                model_name=f"aod_lstm_prediction_entire_data_with_{best_encoded_data.shape[-1]}_features",
                                verbose=0)
    inv_y_actual= label_scaler.inverse_transform(y_scaled)
    inv_y_pred = label_scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    return inv_y_pred

# Run the loop with GRU Seq2Seq reduction model and LSTM prediction model
def predict_pm25_using_grus2s_lstm(X_scaled, y_scaled,
                               label_scaler,
                               range_of_dimension,
                               reduction_n_past, reduction_n_future,
                               reduction_epochs, reduction_batch_size,
                               prediction_n_past, prediction_n_future,
                               prediction_epochs, prediction_batch_size):

    # Run the loop
    loopresults = generate_loopresults(X_scaled, y_scaled,
                                       label_scaler,
                                       range_of_dimension,
                                       LSTMSeq2SeqReduction,
                                       LSTMPrediction,
                                       reduction_n_past, reduction_n_future,
                                       reduction_epochs, reduction_batch_size,
                                       prediction_n_past, prediction_n_future,
                                       prediction_epochs, prediction_batch_size)

    # Get the best data
    best_metrics, best_num_of_components, best_encoded_data, best_evaluation_data, best_encoder_model_path = choose_the_best(loopresults,
                                                                                                                             metric_to_choose="mae",
                                                                                                                             prediction_n_future=prediction_n_future)

    # Predict using encoded data
    y_pred = predictLSTMNoSplit(best_encoded_data, y_scaled,
                                n_past=prediction_n_past, n_future=prediction_n_future,
                                epochs=prediction_epochs, batch_size=prediction_batch_size,
                                model_name=f"aod_lstm_prediction_entire_data_with_{best_encoded_data.shape[-1]}_features",
                                verbose=0)
    inv_y_actual= label_scaler.inverse_transform(y_scaled)
    inv_y_pred = label_scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    return inv_y_pred
    
# Run the loop with CNNLSTM Seq2Seq reduction model and LSTM prediction model
def predict_pm25_using_cnnlstms2s_lstm(X_scaled, y_scaled,
                               label_scaler,
                               range_of_dimension,
                               reduction_n_past, reduction_n_future,
                               reduction_epochs, reduction_batch_size,
                               prediction_n_past, prediction_n_future,
                               prediction_epochs, prediction_batch_size):

    # Run the loop
    loopresults = generate_loopresults(X_scaled, y_scaled,
                                       label_scaler,
                                       range_of_dimension,
                                       LSTMSeq2SeqReduction,
                                       LSTMPrediction,
                                       reduction_n_past, reduction_n_future,
                                       reduction_epochs, reduction_batch_size,
                                       prediction_n_past, prediction_n_future,
                                       prediction_epochs, prediction_batch_size)

    # Get the best data
    best_metrics, best_num_of_components, best_encoded_data, best_evaluation_data, best_encoder_model_path = choose_the_best(loopresults,
                                                                                                                             metric_to_choose="mae",
                                                                                                                             prediction_n_future=prediction_n_future)

    # Predict using encoded data
    y_pred = predictLSTMNoSplit(best_encoded_data, y_scaled,
                                n_past=prediction_n_past, n_future=prediction_n_future,
                                epochs=prediction_epochs, batch_size=prediction_batch_size,
                                model_name=f"aod_lstm_prediction_entire_data_with_{best_encoded_data.shape[-1]}_features",
                                verbose=0)
    inv_y_actual= label_scaler.inverse_transform(y_scaled)
    inv_y_pred = label_scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    return inv_y_pred