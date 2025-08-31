# Author: tphiepbk

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.plot import plot_prediction
from src.config_reader import ConfigurationReader

conf = ConfigurationReader("/le_thanh_van_118/workspace/hiep_workspace/air_quality_index_project/model_params.json").data

def generate_loopresults(range_of_dimension, reduction_model_class, prediction_model_class, X_scaled: pd.DataFrame, y_scaled: pd.DataFrame, label_scaler, with_pm25_3km=True):
    # Get supported metrics
    supported_metrics = prediction_model_class.get_supported_metrics()
    
    # Initialize loopresults
    loopresults = {
        i: {"metrics": {m: None for m in supported_metrics},
            "encoded_data": None,
            "evaluation_data": None,
            "encoder_model_path": None
        }
        for i in range_of_dimension
    }
    
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
        all_days_inv_y_pred, all_days_inv_y_test, metrics = prediction.execute()
    
        # Store results
        for m in supported_metrics:
            loopresults[n]["metrics"][m] = metrics[m][1]
        loopresults[n]["encoded_data"] = X_encoded
        loopresults[n]["encoder_model_path"] = encoder_model_path
        loopresults[n]["evaluation_data"] = (all_days_inv_y_pred, all_days_inv_y_test)

    return loopresults

def choose_the_best(loopresults, metric_to_choose="mae"):
    print("Choosing the best result\n")
    print(loopresults.keys())
    
    # Print out the number of features and corresponding metrics
    #[print(f"N = {n} - mae = {loopresults[n]['mae']}, mse = {loopresults[n]['mse']}, r2 = {loopresults[n]['r2']}") for n in loopresults.keys()]
    for n, loopresult in loopresults.items():
        print(f'N = {n}')
        [print(f"{m} = {loopresult['metrics'][m]}") for m in loopresult["metrics"].keys()]
    
    # Visualize the relation between number of features and MAE
    values = [loopresults[n]["metrics"][metric_to_choose] for n in loopresults.keys()]
    plt.plot(list(loopresults.keys()), values)
    plt.xticks(list(loopresults.keys()))
    plt.xlabel("Number of components")
    plt.yticks(np.arange(min(values), max(values) + 0.1, 0.2))
    plt.ylabel(metric_to_choose)
    plt.show()
    
    # Choose the best number of features
    best_metric_value = 1000
    for n, loopresult in loopresults.items():
        if loopresult["metrics"][metric_to_choose] < best_metric_value:
            best_num_of_components = n
            best_metrics = loopresult["metrics"]
            best_encoded_data = loopresult["encoded_data"]
            best_evaluation_data = loopresult["evaluation_data"]
            best_encoder_model_path = loopresult["encoder_model_path"]
    
    # Visualize the prediction
    plot_prediction(best_evaluation_data[0], best_evaluation_data[1], conf["prediction"]["n_future"])

    return best_metrics, best_num_of_components, best_encoded_data, best_evaluation_data, best_encoder_model_path
