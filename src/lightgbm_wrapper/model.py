# Author: tphiepbk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

import lightgbm as lgb

RANDOM_STATE = 42

# ==========================================================================================

# Reframe past future
def build_supervised_for_horizon(df, horizon_h, target_col):
    # Sort data by station_id then date
    df = df.copy()
    df = df.sort_values(["station_id", "date"])

    # Define target_name, which is the target_col plus horizon_h hour
    target_name = f"{target_col}_t_plus_{horizon_h}h"
    df[target_name] = df.groupby("station_id")[target_col].shift(-horizon_h)

    # Dropna after shifting
    df = df.dropna().reset_index(drop=True)

    # Define label, feature and meta columns
    label_cols = [target_name]
    meta_cols = ["date", "station_id"]
    feature_cols = [c for c in df.columns if c not in label_cols + meta_cols + [target_col]]

    # Get the X, y and meta data
    X = df[meta_cols + feature_cols]
    y = df[meta_cols + label_cols]

    return X, y

# ==========================================================================================

# Split train/test for each station
def train_test_split(X, y, train_ratio=0.8):
    assert (X["date"].equals(y["date"]) and X["station_id"].equals(y["station_id"])), "X and y should have the same metadata values"

    X_train, y_train, X_test, y_test = [], [], [], []
    meta_train, meta_test = [], []

    meta_cols = ["date", "station_id"]
    
    for station in list(sorted(X["station_id"].unique())):
        X_station = X[X["station_id"] == station]
        y_station = y[y["station_id"] == station]
        meta = X_station[meta_cols]

        X_station = X_station.drop(columns=meta_cols)
        y_station = y_station.drop(columns=meta_cols)
    
        n = len(X_station)
        train_end = int(n * train_ratio)
    
        X_train.append(X_station.iloc[:train_end])
        y_train.append(y_station.iloc[:train_end])
        meta_train.append(meta.iloc[:train_end])
    
        X_test.append(X_station.iloc[train_end:])
        y_test.append(y_station.iloc[train_end:])
        meta_test.append(meta.iloc[train_end:])

    X_train = pd.concat(X_train, axis=0)
    X_test = pd.concat(X_test, axis=0)
    
    y_train = pd.concat(y_train, axis=0)
    y_test = pd.concat(y_test, axis=0)
    
    meta_train = pd.concat(meta_train, axis=0)
    meta_test = pd.concat(meta_test, axis=0)

    return (X_train, y_train, meta_train, X_test, y_test, meta_test)

# ==========================================================================================

# Split train/val/test for each station
def train_test_validation_split(X, y, train_ratio=0.7, val_ratio=0.15):
    assert (X["date"].equals(y["date"]) and X["station_id"].equals(y["station_id"])), "X and y should have the same metadata values"

    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    meta_train, meta_val, meta_test = [], [], []

    meta_cols = ["date", "station_id"]
    
    for station in list(sorted(X["station_id"].unique())):
        X_station = X[X["station_id"] == station]
        y_station = y[y["station_id"] == station]
        meta = X_station[meta_cols]

        X_station = X_station.drop(columns=meta_cols)
        y_station = y_station.drop(columns=meta_cols)
    
        n = len(X_station)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
    
        X_train.append(X_station.iloc[:train_end])
        y_train.append(y_station.iloc[:train_end])
        meta_train.append(meta.iloc[:train_end])
    
        X_val.append(X_station.iloc[train_end:val_end])
        y_val.append(y_station.iloc[train_end:val_end])
        meta_val.append(meta.iloc[train_end:val_end])
    
        X_test.append(X_station.iloc[val_end:])
        y_test.append(y_station.iloc[val_end:])
        meta_test.append(meta.iloc[val_end:])

    X_train = pd.concat(X_train, axis=0)
    X_val = pd.concat(X_val, axis=0)
    X_test = pd.concat(X_test, axis=0)
    
    y_train = pd.concat(y_train, axis=0)
    y_val = pd.concat(y_val, axis=0)
    y_test = pd.concat(y_test, axis=0)
    
    meta_train = pd.concat(meta_train, axis=0)
    meta_val = pd.concat(meta_val, axis=0)
    meta_test = pd.concat(meta_test, axis=0)

    return (X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test)

# ==========================================================================================

# Filter dates which have more than 10 records
# Resampled by date, using mean
# Calculate mnbe
def mnbe_avg(y_test_sid, y_pred_sid, meta_test_date_sid):
    threshold = 10
    uniq_date = list(meta_test_date_sid.apply(lambda x: x.date()).unique())
    resampled_y_test = {"date": [], "target": []}
    resampled_y_pred = {"date": [], "target": []}
    #print(f"uniq_date = {uniq_date}")
    for d in uniq_date:
        current_date_meta_test_sid = meta_test_date_sid[meta_test_date_sid.dt.date == d]
        current_date_y_test_sid = y_test_sid.loc[current_date_meta_test_sid.index]
        current_date_y_pred_sid = y_pred_sid.loc[current_date_meta_test_sid.index]
        assert len(current_date_y_test_sid) == len(current_date_y_pred_sid), "current_date_y_test_sid is not equal to current_date_y_pred_sid"
        current_date_num_records = len(current_date_meta_test_sid)
        #print(f"Current date = {d}, number of records = {current_date_num_records}")
        if current_date_num_records >= threshold:
            resampled_y_test["date"].append(d)
            resampled_y_test["target"].append(current_date_y_test_sid.mean())
            resampled_y_pred["date"].append(d)
            resampled_y_pred["target"].append(current_date_y_pred_sid.mean())

    resampled_y_test = pd.DataFrame(resampled_y_test)
    resampled_y_pred = pd.DataFrame(resampled_y_pred)

    #display(resampled_y_test)
    #display(resampled_y_pred)

    #return np.mean((resampled_y_test["target"] - resampled_y_pred["target"]) / resampled_y_test["target"])
    return np.mean((resampled_y_test["target"] - resampled_y_pred["target"]) / resampled_y_test["target"]) * 100.0

# This function receive y_true and y_pred for each station
def compute_metrics(y_true, y_pred, calibrate=False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Convert zero to nan
    yt_nonzero = np.where(y_true == 0, np.nan, y_true)
    
    # MNBE
    #mnbe = np.nanmean((y_true - y_pred) / yt_nonzero)
    if calibrate == True:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(y_pred.reshape(-1,1), yt_nonzero)
        a = reg.intercept_
        b = reg.coef_[0]
        y_pred_corr = a + b * y_pred
        mnbe = np.nanmean((yt_nonzero - y_pred_corr) / yt_nonzero) * 100.0
    else:
        mnbe = np.nanmean((yt_nonzero - y_pred) / yt_nonzero) * 100.0

    # MAPE
    mape = np.nanmean((np.abs((yt_nonzero - y_pred) / yt_nonzero)) * 100.0)

    # Pearson r (with standard deviation check)
    if np.std(y_true) < 1e-6 or np.std(y_pred) < 1e-6:
        r = np.nan
    else:
        r = np.corrcoef(y_true.T, y_pred.T)[0, 1]

    rounded_rmse = round(rmse, 2)
    rounded_mae = round(mae, 2)
    rounded_mape = round(mape, 2)
    rounded_mnbe = round(mnbe, 2)
    rounded_r = round(r, 2)
    
    return {
        #"rmse": rmse,
        #"mae": mae,
        #"mape": mape,
        #"mnbe": mnbe,
        #"r": r,
        "rmse": rounded_rmse,
        "mae": rounded_mae,
        "mape": rounded_mape,
        "mnbe": rounded_mnbe,
        "r": rounded_r,
        "n_samples": len(y_true),
    }
    
# ==========================================================================================

def train_lgbm_for_horizon(df_feat,
                           horizon_h,
                           target_col,
                           learning_rate=0.05,
                           n_estimators=2000,
                           early_stopping_rounds=100,
                           calibrate=False):

    # Build supervised dataset
    X, y = build_supervised_for_horizon(df_feat, horizon_h=horizon_h, target_col=target_col)

    print(f"\nX.shape = {X.shape}")
    print(f"y.shape = {y.shape}")

    print(f"\nNumber of features: {len(X.columns)}")
    print("\nFeatures:")
    print(*list(X.columns), sep="\n")
    print("\nLabel:")
    print(*list(y.columns), sep="\n")

    # Split train validation test
    (X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test) = train_test_validation_split(X, y)
    print(f"\nX_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_val.shape: {X_val.shape}")
    print(f"y_val.shape: {y_val.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")

    # Dataset cho LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": learning_rate,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "device": "cuda",
        "seed": RANDOM_STATE,
    }

    # Early stopping dùng callback (tương thích nhiều version LightGBM)
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=n_estimators,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Prediction
    y_pred = np.expand_dims(model.predict(X_test, num_iteration=model.best_iteration), axis=-1)
    print(f"y_pred.shape: {y_pred.shape}")
    metrics_overall = compute_metrics(y_test, y_pred, calibrate)

    # Metrics for each station
    df_test_res = meta_test.copy()
    df_test_res["y_true"] = y_test.values
    df_test_res["y_pred"] = y_pred

    rows = []
    for sid, grp in df_test_res.groupby("station_id"):
        m = compute_metrics(grp["y_true"], grp["y_pred"], calibrate)
        #print(f"sid = {sid}, grp = {grp}")

        # Calculate avg mnbe
        avg_mnbe = mnbe_avg(grp["y_true"], grp["y_pred"], grp["date"])
        m["avg_mnbe"] = round(avg_mnbe, 2)

        rows.append({"station": int(sid), "horizon_h": horizon_h,**m})
    metrics_by_station = pd.DataFrame(rows)

    return model, metrics_overall, metrics_by_station, (
        X_train, y_train, X_val, y_val, X_test, y_test, meta_test
    )

# ==========================================================================================

def plot_prediction(y_true, y_pred, meta, target_col, title, index_col="date", n_points=300, path=None):
    X_plot = meta.iloc[-n_points:]
    y_true_plot = y_true.iloc[-n_points:]
    y_pred_plot = y_pred[-n_points:]

    plt.figure(figsize=(14, 5))
    plt.plot(X_plot[index_col], y_true_plot, label=f"Actual {target_col}", linewidth=1.5)
    plt.plot(X_plot[index_col], y_pred_plot, label=f"Predicted {target_col}", linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(f"{target_col} (µg/m³)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()
    
def plot_timeseries_example(df, model, horizon_h, station_id, target_col, n_points=300, LIGHTGBM_DIR="."):
    # Prepare data for reference
    X, y = build_supervised_for_horizon(df, horizon_h, target_col)
    (_, _, _,
     _, _, _,
     X_test, y_test, meta_test) = train_test_validation_split(X, y)
    
    # Get the data of station
    mask = (meta_test["station_id"] == station_id)
    X_test_sid = X_test[mask]
    y_test_sid = y_test[mask]
    meta_sid = meta_test[mask]
    
    print(f"X_test_sid.shape: {X_test_sid.shape}")
    print(f"y_test_sid.shape: {y_test_sid.shape}")
    
    if len(X_test_sid) == 0:
        print(f"No test sample for station {station_id}")
        return
    
    # Prediction
    y_pred_sid = model.predict(X_test_sid, num_iteration=getattr(model, "best_iteration", None))
    print(f"y_pred_sid.shape: {y_pred_sid.shape}")

    # Plotting
    plot_prediction(y_test_sid, y_pred_sid, meta_sid, target_col, \
                    title=f"Station {station_id} - Horizon {horizon_h}h", \
                    path=os.path.join(LIGHTGBM_DIR, f"{target_col}_lightgbm_{horizon_h}h_{station_id}_{n_points}.png"))

