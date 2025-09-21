import pandas as pd

from src.config_reader import ConfigurationReader 
from sklearn.metrics import mean_absolute_error

# ==========================================================================================

confReader = ConfigurationReader("/le_thanh_van_118/workspace/hiep_workspace/air_quality_index_project/model_params.json")
conf = confReader.data

# ==========================================================================================

def prepareReducedData(X_scaled, time_indices, *column_name_mapper):
    # Rename columns
    rename_dict = {column: f"aod_feature_{column+1}" for column in X_scaled.columns}
    df_reduced = X_scaled.rename(rename_dict, axis=1)

    # Append additional columns
    for (column, name) in column_name_mapper:
        if type(column) == pd.Series:
            column = column.values
        df_reduced[name] = column
    
    # Set time indices
    df_reduced.set_index(time_indices, inplace=True)
    return df_reduced

# ==========================================================================================

# Augment the reduced AOD data with:
# - pm25_3km values
# - prediced pm25 values
'''
def augmentReducedData(df_aod_reduced, y_aod_scaled,
                       aod_label_scaler,
                      aod_pm25_3km_column):
    # Prepare the reduced data with predicted pm25 values
    # This could be applied only when n_future = 1
    y_pred = predictLSTMNoSplit(df_aod_reduced.drop(columns=["station"], axis=1), y_aod_scaled,
                               n_past=conf["prediction"]["n_past"], n_future=conf["prediction"]["n_future"],
                               epochs=conf["prediction"]["epochs"], batch_size=conf["prediction"]["batch_size"],
                               model_name=f"aod_lstm_prediction_no_split_with_lstms2s_dim_reduction_{df_aod_reduced.shape[-1]}_features",
                               verbose=1)
    inv_y_aod = aod_label_scaler.inverse_transform(y_aod_scaled)
    inv_y_pred = aod_label_scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    hiep_predicted_pm25 = pd.DataFrame(inv_y_pred, columns=["hiep_predicted_pm25"], index=df_aod_reduced.index)
    
    augmented_df_aod_reduced = pd.concat([df_aod_reduced, aod_pm25_3km_column, hiep_predicted_pm25], axis=1)
    
    return augmented_df_aod_reduced
'''
