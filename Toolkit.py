import sys
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def set_working_directory(working_directory):
    """
    Set the working directory for the script.
    
    Parameters:
    working_directory (str): Path to the working directory.
    """
    os.chdir(working_directory)
    print(f"Working directory set to: {working_directory}")

def check_folder(folder):
    """
    Check if a folder exists and if it is empty.
    
    Parameters:
    folder (str): Path to the folder.
    """
    return os.path.exists(folder) and os.listdir(folder)

def filter_tracks(nb_run, filtered_data_folder,nmin=10, len_min=20):
    """
    Filter tracks based on specified criteria.
    
    Parameters:
    nb_run (int): Run number to process.
    nmin (int): Minimum number of neighbors for calculating epsilon.
    len_min (int): Minimum length of tracks to keep.
    """



    if nb_run < 0 or nb_run > 10:
        raise ValueError("Run number must be between 0 and 10.")
    print("Run: ", nb_run)
    run = 'run'+str(nb_run).zfill(2) #add leading zero if needed

    # Load the data
    trackfit_path = 'GVK_CEA_SACLAY_V-A/0.1Hz_anti_'+ run +'_TrackFit.nc' 
    tf =  nc.Dataset(trackfit_path)
    tf_dim_points = tf.dimensions["dim_points"].size

    dr_path = 'DR_Lag_Run_01_to_10/'+run+'_new/Lag/tau11_l10/DR_Drivas_Lag_Full.nc' 
    dr =  nc.Dataset(dr_path)
    dr_dim_points = dr.dimensions["dim_points"].size
    dr_total_number_of_track = dr.dimensions["total_number_of_track"].size

    knn=np.array(dr['N_Neighbors'])
    track_ini=np.array(dr['track_start_index'])-1 # idexes start at 0 in Python instead of 1 in matlab
    track_ini=track_ini.astype(int)
    track_end=np.hstack((np.array(track_ini[1::]-1),dr_dim_points-1))

    iSTB=np.array(dr['Index_from_initial_STB']).astype(int)-1 #idexes start at 0 in Python instead of 1 in matlab
    print("Data loaded")
    # Perform track filtering

    # We avoid samples where epsilon estimated with less than 10 neighbors
    Bad_points=np.where(knn<=nmin)[0]
    Bad_tracks = np.zeros(dr_total_number_of_track, dtype=bool) 
    Bad_tracks[np.searchsorted(track_ini, Bad_points, side='right')-1] = True


    issue_with_index = np.ones(dr_total_number_of_track, dtype=bool)
    for i in range(dr_total_number_of_track-1):
        if track_end[i]-track_ini[i] != iSTB[track_end[i]]-iSTB[track_ini[i]]:
            issue_with_index[i] = False
    
    # Array of lengths of tracks
    len_tracks=track_end-track_ini +1

    # mask of good tracks

    #keep only the tracks that are not bad, have a length greater than len_min and have no issue with the index
    mask_dr = ~Bad_tracks & (len_tracks > len_min) & issue_with_index 
    track_ini_good = track_ini[mask_dr]
    track_end_good = track_end[mask_dr]

    #new index of good tracks
    track_start_index_good = np.zeros(len(track_ini_good), dtype=int)
    track_start_index_good[0] = 0
    for i in range(1,len(track_ini_good)):
        track_start_index_good[i] = track_start_index_good[i-1] + (track_end_good[i-1] - track_ini_good[i-1] + 1)

    # Create a mask for the filtered tracks in the tf file
    mask_tf = np.zeros(tf_dim_points, dtype=bool)
    mask_Drivas = np.zeros(dr_dim_points, dtype=bool)

    # Loop over the good tracks
    for i in range(len(track_ini_good)):
        # Get the start and end indices of the track
        dr_start = track_ini_good[i]
        dr_end = track_end_good[i]
        mask_Drivas[dr_start:dr_end+1] = True

        # Get the corresponding indices in the tf file
        tf_start = iSTB[dr_start]
        tf_end = iSTB[dr_end]
        mask_tf[tf_start:tf_end+1] = True 

    print("Tracks filtered")

    # Save the filtered data to a new NetCDF file
    print("Saving filtered data to NetCDF file...")
    if not os.path.exists(filtered_data_folder):
        os.makedirs(filtered_data_folder)
    folder_path = os.path.join(filtered_data_folder, f"{run}_filtered.nc")
    with nc.Dataset(folder_path, 'w', format='NETCDF4') as ds:
        # Create dimensions
        ds.createDimension('dim_points', np.sum(mask_tf))
        ds.createDimension('total_number_of_track', np.sum(mask_dr))

        # Create variables
        variables = ['x', 'y', 'z', 'u', 'v', 'w', 'ax', 'ay', 'az']
        for var in variables:
            ds.createVariable(var, 'f4', ('dim_points',))

        ds.createVariable('Drivas', 'f4', ('dim_points',))
        ds.createVariable('track_start_index', 'f4', ('total_number_of_track',))
        
        for var in variables:
            ds[var][:] = np.asarray(tf[var])[mask_tf]
            print(var, "filtered")

        ds['Drivas'][:] = np.asarray(dr['DR_Drivas'])[mask_Drivas]
        print("Drivas filtered")
        ds['track_start_index'][:] = np.asarray(track_start_index_good, dtype='f4')
        print("track_start_index written")

    print("Data saved to", folder_path)

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

def split_data(nb_run,p, filtered_data_folder, split_data_folder):

    """
    Splits the data into smaller segments for processing.
    
    parameters:
    nb_run (int): Run number to process.
    p (int): Size of the segments.
    filtered_data_folder (str):Folder containing filtered data.
    split_data_folder (str): Folder where processed data will be saved.
    """
    
    if os.path.exists(os.path.join(split_data_folder, f"p={p}")):
        print(f"P={p} already processed. Skipping run {nb_run}.")
        return
    var = ['v', 'u', 'w','ax', 'ay', 'az','Drivas']
    run = "run" + str(nb_run).zfill(2)
    print("Run: ", nb_run)

    # Load the data
    print("Loading data...")
    
    trackfit_path = os.path.join(filtered_data_folder, f"{run}_filtered.nc")
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
    if not os.path.exists(split_data_folder):
        os.makedirs(split_data_folder)

    if not os.path.exists(os.path.join(split_data_folder, f"p={p}")):
        os.makedirs(os.path.join(split_data_folder, f"p={p}"))

    parquet_path = os.path.join(split_data_folder,os.path.join(f"p={p}", f"{run}_p={p}.parquet"))
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

def load_database(nb_run,p,split_data_folder,filtered_data_folder):
    """
    Opens data from a specified path and if it does not exist, it creates the folder and splits the data with the given parameters.
    
    Parameters:
    run (int): Run number for data selection.
    p (int): Size of the segments.
    processed_data_folder (str): Folder where processed data is stored.
    filtered_data_folder (str): Folder where filtered data is stored.
    
    Returns:
    numpy.ndarray: The loaded data as a NumPy array.
    """
    run = str(nb_run).zfill(2)
    path = os.path.join(split_data_folder,os.path.join(f"p={p}", f"run{run}_p={p}.parquet"))
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        print("Creating the folder and splitting the data...")
        split_data(nb_run, p, filtered_data_folder, split_data_folder)
        print("Data split and saved.")
    data = pd.read_parquet(path, engine="pyarrow")
    return data.to_numpy()

def prediction_neighbors(db,ratio_train, k, weighting=True, normalize=False):

    """
    Predicts future values using a weighted nearest neighbors approach.
    
    Parameters:
    db (array-like): Input data series.
    ratio_train (float): Ratio of training data to total data.
    k (int): Number of neighbors to consider.
    weighting (bool): Whether to apply exponential distance-based weighting.
    normalize (bool): Whether to normalize input sequences before processing.
    
    Returns:
    tuple: Predicted values and associated weighted covariance.
    """
    if ratio_train <= 0 or ratio_train >= 1:
        raise ValueError("ratio_train must be between 0 and 1.")
    db_train = db[:int(db.shape[0]*ratio_train)]
    db_test = db[int(db.shape[0]*ratio_train):]

    _, pp1 = db_train.shape
    p = pp1// 7

    mask = [False if i % p == (p-1)  else True for i in range(pp1-p)] # mask for features
    mask += [True if i % p == (p-1) else False for i in range(p)] # mask for the Drivas 

    # if normalize:
    #     puplet_means = np.mean(puplets, axis=1)
    #     puplets -= puplet_means[:, np.newaxis]
    #     db_means = np.mean(db[:, :-1], axis=1)
    #     db -= db_means[:, np.newaxis]

    db_train = db_train[:, mask]
    db_test = db_test[: , mask]

    print("Training model...")
    neigh = NearestNeighbors(n_jobs=4)
    neigh.fit(db_train[:, :-1])
    dist, idx = neigh.kneighbors(db_test[:, :-1], n_neighbors=k, return_distance=True)

    med = np.median(dist, axis=1)

    if weighting and np.min(med) != 0:
        weights = np.exp(-dist / med[:, np.newaxis])
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
    else:
        weights = np.ones_like(dist)

    vals = np.full_like(db_test[:,-1], np.nan)
    #weighted_covariance = np.zeros_like(vals)

    print("Calculating coefficients...")
    x = db_train[idx, :-1]
    y = (weights * db_train[idx, -1])[:, :, np.newaxis]
    
    x_test = db_test[:, :-1]
    y_test = db_test[:, -1]

    x = np.pad(x, [(0, 0), (0, 0), (1, 0)], mode='constant', constant_values=1) 
    x_test = np.pad(x_test, [(0, 0), (1, 0)], mode='constant', constant_values=1)
    coef = pinv(np.transpose(x, axes=[0, 2, 1]) @ (weights[:, :, np.newaxis] * x)) @ np.transpose(x, axes=[0, 2, 1]) @ y
    coef = np.squeeze(coef)

    vals = np.sum(coef * x_test, axis=1)

    #residuals = db_train[idx, -1] - np.sum(coef * np.transpose(x, (0, 2, 1)), axis=1)
    #weighted_covariance = (np.sum(weights * residuals**2, axis=1) / np.sum(weights, axis=1))

    # if normalize:
    #     vals[(p - 1) * stride + 1:] += puplet_means[:-1]
    return vals , y_test

def stats_results(y_predicted,y):
    """
    Calculate various performance metrics for predicted values.
    
    Parameters:
    y_predicted (array-like): Predicted values.
    y (array-like): Actual values.
    
    Returns:
    dict: Dictionary containing various performance metrics.
    """
    # Calculate the mean squared error
    mse = mean_squared_error(y, y_predicted)
    
    # Calculate the root mean squared error
    rmse = np.sqrt(mse)
    
    # Calculate the correlation coefficient
    corr = np.corrcoef(y, y_predicted)[0, 1]
    
    # Calculate the mean absolute error
    mae = np.mean(np.abs(y - y_predicted))

    # Calculate the R-squared value
    ss_res = np.sum((y - y_predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)


    return [mse, rmse, corr, mae, r_squared]
        