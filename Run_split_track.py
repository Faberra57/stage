import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import Toolkit as tk

if __name__ == "__main__":
    # Set the working directory and run
    working_directory = "/Volumes/T7 Shield/ALEAS/"
    os.chdir(working_directory)
    run_training = ['run01', 'run02', 'run03', 'run04', 'run05']
    var = ['v', 'u', 'w','ax', 'ay', 'az','Drivas']
    p = 3 #size of the tuples
    for run in run_training:
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
                data = tk.create_database(data, p, overlapping=False)
                db_3[j, last_line_filled:last_line_filled+data.shape[0],:] = data.astype(float)
            last_line_filled += data.shape[0]

        db_3 = db_3[:,:last_line_filled,:] #remove the empty lines
        print("Saving the data...")
        columns = [f"{var[j]}_{k}" for j in range(len(var)) for k in range(p)]
        db_3_reshaped = db_3.reshape(db_3.shape[1], -1) 
        # an example of the columns:
        # v_0   v_1   v_2   u_0   u_1   u_2   w_0   w_1   w_2   ax_0  ax_1  ax_2   ay_0  ay_1  ay_2   az_0  az_1  az_2   Drivas_0  Drivas_1  Drivas_2
        df = pd.DataFrame(db_3_reshaped,columns=columns)
        parquet_path = os.path.join("Slipt_Tracks", f"{run}_{p}.parquet")
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

