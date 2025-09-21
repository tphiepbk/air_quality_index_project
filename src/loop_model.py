# Author: tphiepbk

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.plot import plot_prediction

def generate_loopresults(X_scaled: pd.DataFrame, y_scaled: pd.DataFrame,
                         label_scaler,
                         range_of_dimension,
                         reduction_model_class,
                         prediction_model_class,
                         reduction_model_name_prefix="reduction_model_prefix",
                         prediction_model_name_prefix="prediction_model_prefix",
                         reduction_n_past=7, reduction_n_future=7,
                         reduction_epochs=50, reduction_batch_size=128,
                         prediction_n_past=7, prediction_n_future=1,
                         prediction_epochs=50, prediction_batch_size=128,
                         saved_model_weight_dir=".", saved_model_plot_dir=".",
                         with_pm25_3km=False):
    # Get supported metrics
    supported_metrics = prediction_model_class.get_supported_metrics()
    
    # Initialize loopresults
    loopresults = {
        i: {"metrics": {m: None for m in supported_metrics},
            "encoded_data": None,
            "evaluation_data": None,
            "encoder_model_path": None,
            "prediction_model_path": None,
        }
        for i in range_of_dimension
    }
    
    # Loop between min and (number of features - 1) to choose what number is the best
    for n in range_of_dimension:
        reduction_model_name = f"{reduction_model_name_prefix}_{reduction_model_class.get_class_name()}_{n}_features{'with_pm25_3km' if with_pm25_3km == True else ''}"
        if with_pm25_3km:
            reduction_model_name = reduction_model_name + "with_pm25_3km_as_feature"
        # Apply Seq2seq
        reduction_model = reduction_model_class(X_scaled,
                                        val_percentage=0.2, test_percentage=0.2,
                                        latent_dim=n,
                                        n_past=reduction_n_past, n_future=reduction_n_future,
                                        epochs=reduction_epochs, batch_size=reduction_batch_size,
                                        verbose=1,
                                        model_name=reduction_model_name)
        X_encoded, reduction_model_path, encoder_model_path = reduction_model.execute(saved_model_weight_dir)
        reduction_model.dump(saved_model_plot_dir)
        reduction_model.dump_encoder(saved_model_plot_dir)
        
        # Prediction
        prediction_model_name = f"{prediction_model_name_prefix}_{prediction_model_class.get_class_name()}_with_{reduction_model_class.get_class_name()}_{n}_features"
        if with_pm25_3km:
            prediction_model_name = prediction_model_name + "with_pm25_3km_as_feature"
        prediction_model = prediction_model_class(X_encoded, y_scaled,
                                            label_scaler,
                                            val_percentage=0.2, test_percentage=0.2,
                                            n_past=prediction_n_past, n_future=prediction_n_future,
                                            epochs=prediction_epochs, batch_size=prediction_batch_size,
                                            model_name=prediction_model_name,
                                            verbose=1)
        prediction_model.dump(saved_model_plot_dir)
        all_days_inv_y_pred, all_days_inv_y_test, metrics, prediction_model_path = prediction_model.execute(saved_model_weight_dir)
    
        # Store results
        for m in supported_metrics:
            loopresults[n]["metrics"][m] = metrics[m][1]
        loopresults[n]["encoded_data"] = X_encoded
        loopresults[n]["encoder_model_path"] = encoder_model_path
        loopresults[n]["prediction_model_path"] = prediction_model_path
        loopresults[n]["evaluation_data"] = (all_days_inv_y_pred, all_days_inv_y_test)

    return loopresults

def choose_the_best(loopresults, metric_to_choose="mae", prediction_n_future=1):
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
            best_prediction_model_path = loopresult["prediction_model_path"]
            best_metric_value = loopresult["metrics"][metric_to_choose]

    # Visualize the prediction
    plot_prediction(best_evaluation_data[0], best_evaluation_data[1], prediction_n_future)

    return best_metrics, best_num_of_components, best_encoded_data, best_evaluation_data, best_encoder_model_path, best_prediction_model_path
