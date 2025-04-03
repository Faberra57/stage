import netCDF4 as nc
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":  
    working_directory = "/Volumes/T7 Shield/ALEAS/"
    os.chdir(working_directory)
    nb_runs = 11

    results = []  

    for i in range(1, nb_runs):
        print("Run:", i)
        run = 'run' + str(i).zfill(2)  

        # Load the data
        trackfit_path = f'Filtered_Tracks/{run}_filtered.nc'
        file = nc.Dataset(trackfit_path)

        dim_points = file.dimensions["dim_points"].size
        total_number_of_track = file.dimensions["total_number_of_track"].size

        results.append([run, dim_points, total_number_of_track])


    df = pd.DataFrame(results, columns=["Run", "Dim Points", "Nb Tracks"])
    print(df)