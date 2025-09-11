import pandas as pd
import numpy as np

# Read the CSV file to Pandas Dataframe
def csv_to_dataframe(csvpath):
    # Read CSV file
    df = pd.read_csv(csvpath, index_col=None)
    # Lower case all columns
    df = df.rename(columns={name: name.lower() for name in df.columns})
    # Convert "time" columns to Pandas datetime
    df = df.assign(time=pd.to_datetime(df["time"]))
    # Sort data by "station" then "time"
    df.sort_values(by=["station", "time"], ascending=[True, True], inplace=True)
    # Set "time" column as index
    df.set_index("time", inplace=True)
    # Convert the -9999 to nan
    df[df <= -9999] = np.nan
    # Return
    return df
