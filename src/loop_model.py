# Author: tphiepbk

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.plot import plot_prediction
from src.config_reader import ConfigurationReader

conf = ConfigurationReader("/le_thanh_van_118/workspace/hiep_workspace/air_quality_index_project/model_params.json").data

def generate_loopresults(range_of_dimension, reduction_model_class, prediction_model_class, X_scaled: pd.DataFrame, y_scaled: pd.DataFrame, label_scaler, with_pm25_3km=True):
    # Initialize loopresults
    loopresults = {i:{"mae": None, "encoded_data": None, "evaluation_data": None, "encoder_model_path": None} for i in range_of_dimension}
    
    # Loop between min and (number of features - 1) to choose what number is the best
    for n in range_of_dimension:
        reduction_model_name = f"aod_{reduction_model_class.get_class_name()}_{n}_features{'_no_pm25_3km' if with_pm25_3km == False else ''}"
        # Apply Seq2seq
        reduction = reduction_model_class(X_scaled,
                                        val_percentage=0.2, test_percentage=0.2,
                                        latent_dim=n,
                                        n_past=conf["reduction"]["n_past"], n_future=conf["reduction"]["n_future"],
                                        epochs=conf["reduction"]["epochs"], batch_size=conf["reduction"]["batch_size"],
                                        verbose=0,
                                        model_name=reduction_model_name)
        X_encoded, encoder_model_path = reduction.execute()
        reduction.get_model_info()
        reduction.get_encoder_model_info()
        
        # Prediction
        prediction_model_name = f"aod_{prediction_model_class.get_class_name()}_with_{reduction_model_class.get_class_name()}_{n}_features{'_no_pm25_3km' if with_pm25_3km == False else ''}"
        prediction = prediction_model_class(X_encoded, y_scaled,
                                            label_scaler,
                                            val_percentage=0.2, test_percentage=0.2,
                                            n_past=conf["prediction"]["n_past"], n_future=conf["prediction"]["n_future"],
                                            epochs=conf["prediction"]["epochs"], batch_size=conf["prediction"]["batch_size"],
                                            model_name=prediction_model_name,
                                            verbose=0)
        prediction.get_model_info()
        all_days_inv_y_pred, all_days_inv_y_test, all_days_mae, avg_mae, all_days_mse, avg_mse, all_days_r2, avg_r2 = prediction.execute()
    
        # Store results
        loopresults[n]["mae"] = avg_mae
        loopresults[n]["mse"] = avg_mse
        loopresults[n]["r2"] = avg_r2
        loopresults[n]["encoded_data"] = X_encoded
        loopresults[n]["encoder_model_path"] = encoder_model_path
        loopresults[n]["evaluation_data"] = (all_days_inv_y_pred, all_days_inv_y_test)

    return loopresults

def choose_the_best(loopresults, metric="mae"):
    assert metric in ["mae", "mse", "r2"], "Metric to choose the best dimension should be: mae, mse, r2"
    
    # Print out the number of features and corresponding MAE
    [print(f"N = {n} - mae = {loopresults[n]['mae']}, mse = {loopresults[n]['mse']}, r2 = {loopresults[n]['r2']}") for n in loopresults.keys()]
    
    # Visualize the relation between number of features and MAE
    values = [loopresults[n][metric] for n in loopresults.keys()]
    plt.plot(list(loopresults.keys()), values)
    plt.xticks(list(loopresults.keys()))
    plt.xlabel("Number of components")
    plt.yticks(np.arange(min(values), max(values) + 0.1, 0.2))
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.show()
    
    # Choose the best number of features
    best_metric_value = 1000
    for n in loopresults.keys():
        if loopresults[n][metric] < best_metric_value:
            best_num_of_components = n
            best_mae = loopresults[n]["mae"]
            best_mse = loopresults[n]["mse"]
            best_r2 = loopresults[n]["r2"]
            best_encoded_data = loopresults[n]["encoded_data"]
            best_evaluation_data = loopresults[n]["evaluation_data"]
            best_encoder_model_path = loopresults[n]["encoder_model_path"]
    
    # Visualize the prediction
    plot_prediction(best_evaluation_data[0], best_evaluation_data[1], conf["prediction"]["n_future"])

    return best_mae, best_mse, best_r2, best_num_of_components, best_encoded_data, best_evaluation_data, best_encoder_model_path
