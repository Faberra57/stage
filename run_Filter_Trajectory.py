import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    working_directory = "/Volumes/T7 Shield/ALEAS/"
    # Change working directory
    os.chdir(working_directory)
    nb_runs = 11

    # parameters for the filtering
    nmin=10
    len_min=20

    for i in range(1,nb_runs):
        print("Run: ", i)
        run = 'run'+str(i).zfill(2) #add leading zero if needed

        # Load the data
        trackfit_path = 'GVK_CEA_SACLAY_V-A/0.1Hz_anti_'+ run +'_TrackFit.nc' 
        tf =  nc.Dataset(trackfit_path)
        tf_dim_points = tf.dimensions["dim_points"].size
        tf_total_number_of_track = tf.dimensions["total_number_of_track"].size


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
        output_path = 'Filtered_Tracks/' + run + '_filtered.nc'
        with nc.Dataset(output_path, 'w', format='NETCDF4') as ds:
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

        print("Data saved to", output_path)
        