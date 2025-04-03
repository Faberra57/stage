import sys
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def set_working_directory(working_directory):
    """
    Set the working directory for the script.
    
    Parameters:
    working_directory (str): Path to the working directory.
    """
    os.chdir(working_directory)
    print(f"Working directory set to: {working_directory}")

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

def split_data(working_directory,processed_data_folder,run_list,p):

    """
    Splits the data into smaller segments for processing.
    
    parameters:
    working_directory (str): Path to the working directory. (e.g. "/Volumes/T7 Shield/ALEAS/")
    processed_data_folder (str): Folder for processed data.
    run_list (list): List of run numbers to process.
    p (int): Size of the tuples.
    """


    # Set the working directory and run
    os.chdir(working_directory)
    var = ['v', 'u', 'w','ax', 'ay', 'az','Drivas']
    for run in run_list:
        print("Run: ", run)
        # Load the data
        print("Loading data...")
        trackfit_path = f'Filtered_Tracks/{run}_filtered.nc'
        file = nc.Dataset(trackfit_path)
        total_number_of_track = file.dimensions["total_number_of_track"].size
        dim_point = file.dimensions["dim_points"].size
        dim_point = int(dim_point)
        track_ini = np.array(file.variables["track_start_index"])
        track_ini = track_ini.astype(int)

        #matrice with raw data
        var = np.array(var)
        db_2 = np.zeros((len(var),dim_point)) #database with raw data with 2 dimensions
        for i,var_name in enumerate(var):
            data = np.array(file.variables[var_name])  # Charger les données de la variable
            db_2[i, :] = data.astype(float)


        print("Splitting the tracks...")
        db_3 = np.zeros((len(var),(dim_point//p)+1,p)) #(dim_point//p)+1 for not being to memory expensive and reshape it later
        # df_3[i,j,k]  
        # i -> feature (v, u, w, ax, ay, az, Drivas)
        # j -> list of p values of the feature 
        # k -> point of the list of p values of the feature 

        #slipt the track

        #Because the size of each track is not divisible by p, 
        #we need to keep track of the last filled line in db_3
        last_line_filled = 0 

        for i in range(total_number_of_track-1):
            start_track = track_ini[i]
            end_track = track_ini[i+1] 
            for j, var_name in enumerate(var):
                data = db_2[j, start_track:end_track]
                data = create_database(data, p, overlapping=False)
                db_3[j, last_line_filled:last_line_filled+data.shape[0],:] = data.astype(float)
            last_line_filled += data.shape[0]

        db_3 = db_3[:,:last_line_filled,:] #remove the empty lines
        print("Saving the data...")
        columns = [f"{var[j]}_{k}" for j in range(len(var)) for k in range(p)]
        db_3_reshaped = db_3.reshape(db_3.shape[1], -1) 
        # an example of the columns:
        # v_0   v_1   v_2   u_0   u_1   u_2   w_0   w_1   w_2   ax_0  ax_1  ax_2   ay_0  ay_1  ay_2   az_0  az_1  az_2   Drivas_0  Drivas_1  Drivas_2
        df = pd.DataFrame(db_3_reshaped,columns=columns)
        if not os.path.exists(processed_data_folder):
            os.makedirs(processed_data_folder)
        parquet_path = os.path.join(processed_data_folder, f"{run}_p={p}.parquet")
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

def prediction_neighbors(data, db, k, stride=1, weighting=True, normalize=False):
    """
    Predicts future values using a weighted nearest neighbors approach.
    
    Parameters:
    data (ndarray): Input time series data.
    db (ndarray): Precomputed database of past-value sequences.
    k (int): Number of nearest neighbors to consider.
    stride (int): Step size between samples.
    weighting (bool): Whether to apply exponential distance-based weighting.
    normalize (bool): Whether to normalize input sequences before processing.
    
    Returns:
    tuple: Predicted values and associated weighted covariance.
    """
    np_, pp1 = db.shape
    p = pp1 - 1
    
    puplets = np.lib.stride_tricks.sliding_window_view(data, window_shape=(stride * (p - 1) + 1,))[:, ::stride].copy()
    
    if normalize:
        puplet_means = np.mean(puplets, axis=1)
        puplets -= puplet_means[:, np.newaxis]
        db_means = np.mean(db[:, :-1], axis=1)
        db -= db_means[:, np.newaxis]
    
    neigh = NearestNeighbors(n_jobs=4)
    neigh.fit(db[:, :-1])
    dist, idx = neigh.kneighbors(puplets, n_neighbors=k, return_distance=True)
    
    med = np.median(dist, axis=1)
    
    if weighting and np.min(med) != 0:
        weights = np.exp(-dist / med[:, np.newaxis])
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
    else:
        weights = np.ones_like(dist)
    
    vals = np.full_like(data, np.nan)
    weighted_covariance = np.zeros_like(vals)
    
    x = db[idx, :-1]
    y = (weights * db[idx, -1])[:, :, np.newaxis]
    
    x = np.pad(x, [(0, 0), (0, 0), (1, 0)], mode='constant', constant_values=1)
    coef = pinv(np.transpose(x, axes=[0, 2, 1]) @ (weights[:, :, np.newaxis] * x)) @ np.transpose(x, axes=[0, 2, 1]) @ y
    vals[(p - 1) * stride + 1:] = coef[:-1, 0, 0] + np.sum(coef[:, 1:, 0] * puplets, axis=1)[:-1]
    
    residuals = db[idx, -1] - np.sum(coef * np.transpose(x, (0, 2, 1)), axis=1)
    weighted_covariance[(p - 1) * stride + 1:] = (np.sum(weights * residuals**2, axis=1) / np.sum(weights, axis=1))[:-1]
    
    if normalize:
        vals[(p - 1) * stride + 1:] += puplet_means[:-1]
    
    return vals, weighted_covariance

def open_data(folder,run,p):
    """
    Opens data from a specified path.
    
    Parameters:
    path (str): Path to the data folder.
    run (int): Run number for data selection.
    
    Returns:
    DataFrame with data.
    """
    run = str(run).zfill(2)
    path = os.path.join(folder, f"run{run}_p={p}.parquet")
    data = pd.read_parquet(path, engine="pyarrow")
    return data
