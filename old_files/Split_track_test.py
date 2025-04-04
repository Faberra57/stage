#%%

import netCDF4 as nc
import numpy as np
import pandas as pd
import os

def create_database(data, p, overlapping=False):
    """
    Create a database of p continuous values with or without overlapping.

    Parameters:
    data (array-like): Input data series.
    p (int): The size of the tuples.
    overlapping (bool): Whether the tuples should overlap.

    Returns:
    np.ndarray: The database of tuples.
    """
    # Create sliding window view of the data with window size p+1
    window = np.lib.stride_tricks.sliding_window_view(data, window_shape=(p,))
    # Set stride based on overlapping flag
    stride = 1 if overlapping else p
    # Select windows based on the stride
    db = window[0::stride, :]
    return db

#%% 

working_directory = "/Volumes/T7 Shield/ALEAS/"
os.chdir(working_directory)
run = 'run01'

var = ['v', 'u', 'w','ax', 'ay', 'az','Drivas']
trackfit_path = f'Filtered_Tracks/{run}_filtered.nc'

# Load the data
file = nc.Dataset(trackfit_path)
total_number_of_track = file.dimensions["total_number_of_track"].size
dim_point = file.dimensions["dim_points"].size
dim_point = int(dim_point)
track_ini = np.array(file.variables["track_start_index"])
track_ini = track_ini.astype(int)

#%%
#matrice with raw data
var = np.array(var)
db_2 = np.zeros((len(var),dim_point)) #database with raw data with 2 dimensions
for i,var_name in enumerate(var):
    data = np.array(file.variables[var_name])  # Charger les données de la variable
    db_2[i, :] = data.astype(float)

#%%

db_3 = np.zeros((len(var),(dim_point//3)+1,3)) #(dim_point//3)+1 for not being to memory expensive and reshape it later
# df_3[i,j,k]  
# i -> feature (v, u, w, ax, ay, az, Drivas)
# j -> list of 3 values of the feature 
# k -> point of the list of 3 values of the feature 


#slipt the track

#Because the size of each track is not divisible by 3, 
#we need to keep track of the last filled line in db_3
last_line_filled = 0 

for i in range(total_number_of_track-1):
    start_track = track_ini[i]
    end_track = track_ini[i+1] 
    for j, var_name in enumerate(var):
        data = db_2[j, start_track:end_track]
        data = create_database(data, 3, overlapping=False)
        db_3[j, last_line_filled:last_line_filled+data.shape[0],:] = data.astype(float)
    last_line_filled += data.shape[0]

db_3 = db_3[:,:last_line_filled,:] #remove the empty lines

#%%
columns = [f"{var[j]}_{k}" for j in range(len(var)) for k in range(3)]
db_3_reshaped = db_3.reshape(db_3.shape[1], -1)  # (dim_points/3, nb_var * 3)
df = pd.DataFrame(db_3_reshaped,columns=columns)

parquet_path = os.path.join("Slipt_Tracks", f"{run}_db_3.parquet")
df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")