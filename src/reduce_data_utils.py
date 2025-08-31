import pandas as pd

from src.config_reader import ConfigurationReader 
from src.prediction_model.lstm import predictLSTMNoSplit
from sklearn.metrics import mean_absolute_error

# ==========================================================================================

confReader = ConfigurationReader("/le_thanh_van_118/workspace/hiep_workspace/air_quality_index_project/model_params.json")
conf = confReader.data

# ==========================================================================================

def prepareReducedData(X_scaled, time_indices, station_column=None):
    # Deep copy
    df_aod_reduced = X_scaled.copy(deep=True)

    # Rename columns
    rename_dict = {column: f"aod_feature_{column+1}" for column in df_aod_reduced.columns}
    df_aod_reduced.rename(rename_dict, axis=1, inplace=True)
    
    # Set station column
    if station_column is not None:
        df_aod_reduced["station"] = station_column.values

    # Set time indices
    df_aod_reduced.set_index(time_indices, inplace=True)
    return df_aod_reduced

# ==========================================================================================

# Augment the reduced AOD data with:
# - pm25_3km values
# - prediced pm25 values
def augmentReducedData(df_aod_reduced, y_aod_scaled, aod_pm25_3km_column, aod_label_scaler):
    # Prepare the reduced data with predicted pm25 values
    # This could be applied only when n_future = 1
    y_pred = predictLSTMNoSplit(df_aod_reduced.drop(columns=["station"], axis=1), y_aod_scaled,
                               n_past=conf["prediction"]["n_past"], n_future=conf["prediction"]["n_future"],
                               epochs=conf["prediction"]["epochs"], batch_size=conf["prediction"]["batch_size"],
                               model_name=f"aod_lstm_prediction_no_split_with_lstms2s_dim_reduction_{df_aod_reduced.shape[-1]}_features",
                               verbose=0)
    inv_y_aod = aod_label_scaler.inverse_transform(y_aod_scaled)
    inv_y_pred = aod_label_scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    hiep_predicted_pm25 = pd.DataFrame(inv_y_pred, columns=["hiep_predicted_pm25"], index=df_aod_reduced.index)
    
    augmented_df_aod_reduced = pd.concat([df_aod_reduced, aod_pm25_3km_column, hiep_predicted_pm25], axis=1)
    
    return augmented_df_aod_reduced
