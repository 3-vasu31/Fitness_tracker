import pandas as pd
from glob import glob


files = glob("../../data/raw/MetaMotion/*.csv")
data_path = "../../data/raw/MetaMotion\\"
def read_data_from_files(files):
    """ Reads data from CSV files, extracts participant, label, and category information,
        and returns two DataFrames: one for accelerometer data and one for gyroscope data.
        
        Args:
            files (list): List of file paths to CSV files.
        
        Returns:
            acc_df (DataFrame): DataFrame containing accelerometer data.
            gyr_df (DataFrame): DataFrame containing gyroscope data.
    """
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1 #unique participant id
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
            
        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df], ignore_index=True)

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df], ignore_index=True)
            
    # converting epochs (ms) as index for the dataframe to convert it into time serites dataframe
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")

    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # now deleting rest time related columns

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files)

# Merging datasets

data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis = 1)

# data_merged.dropna() // this shows that out of about 70,000 rows there are only about 1000 rows which have both accelerometer and gyroscope data.

'''Since we are using two sensors to collect data and the gyroscope collects data in double the freq, and we are merging these columns on the index there are many rows which are missing. there are some time frames for which the milli seconds match and we get the data for both gyroscope and accelerometer.'''
# data_merged.dropna() 

# Resample data (frequency conversion)
# since Gyroscope is at 25Hz and Accelerometer is at 12.5Hz, gyroscope collects more data in one second than accelerometer.
# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

#Custom aggrigation function to resample data: For numerical columns apply mean, for categorical columns apply last value
rename_map = {
    'x-axis (g)': "acc_x",
    'y-axis (g)': "acc_y",
    'z-axis (g)': "acc_z",
    'x-axis (deg/s)': "gyr_x",
    'y-axis (deg/s)': "gyr_y",
    'z-axis (deg/s)': "gyr_z",
    'participant': "participant",
    'label': "label",
    'category': "category",
    'set': "set"
}

data_merged.rename(columns=rename_map, inplace=True)
sampling = {
    "acc_x" : "mean",
    "acc_y" : "mean",
    "acc_z" : "mean",
    "gyr_x" : "mean",
    "gyr_y" : "mean",
    "gyr_z" : "mean",
    "participant" : "last",  
    "label" : "last",
    "category" : "last",
    "set" : "last"
}
# Resampling the data to 200ms
# This is done to reduce the size of the data and also to make it compatible with the model we are using.

data_merged[:100].resample(rule = "200mS").apply(sampling)

# split the data by day to save the machine computation and since we have data for whole week and sampling in mili seconds the data will explode in size
# n and g are the groupby variables, n is the name of the group and g is the data in that group
days = [g for n, g in data_merged.groupby(pd.Grouper(freq = "D"))]

data_resampled = pd.concat([df.resample(rule = "200ms").apply(sampling).dropna() for df in days])

data_resampled.info()

data_resampled["set"] = data_resampled["set"].astype(int)

# Export dataset
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")


