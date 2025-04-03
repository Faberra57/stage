#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%% Import Libraries
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

#%% Read Data

import os

working_directory = "/Volumes/T7 Shield/ALEAS/"
# Change working directory
os.chdir(working_directory)


run ='run01'

test = True # if True it will run some tests


trackfit_path = 'GVK_CEA_SACLAY_V-A/0.1Hz_anti_'+run+'_TrackFit.nc' 
tf =  nc.Dataset(trackfit_path)
tf_dim_points = tf.dimensions["dim_points"].size
tf_total_number_of_track = tf.dimensions["total_number_of_track"].size


dr_path = 'DR_Lag_Run_01_to_10/'+run+'_new/Lag/tau11_l10/DR_Drivas_Lag_Full.nc' 
dr =  nc.Dataset(dr_path)
dr_dim_points = dr.dimensions["dim_points"].size
dr_total_number_of_track = dr.dimensions["total_number_of_track"].size

#%% First we look for the tracks with epsilon estimated with more than 10 neighbors Step 1/2

knn=np.array(dr['N_Neighbors'])

nmin=10 # Minimal number of neighbors

# Definition of index of initial and final position of tracks (-1 for Python)
track_ini=np.array(dr['track_start_index'])-1
track_ini=track_ini.astype(int)

track_end=np.hstack((np.array(track_ini[1::]-1),dr_dim_points-1))


#track_end=track_end.astype(int)

# We avoid samples where epsilon estimated with less than 10 neighbors
Bad_points=np.where(knn<=nmin)[0]
#%% First we look for the tracks with epsilon estimated with more than 10 neighbors Step 2/2

# Bad_tracks=np.zeros(len(track_ini))


# Initialization of bad_tracks with False
Bad_tracks = np.zeros(dr_total_number_of_track, dtype=bool) 

# # Identification of 'bad' tracks with less than 10 neighbors (1 means bad track)
# for i in range(len(Bad_points)):
#     print(i,'  of  ',len(Bad_points))
#     #check1=np.amax(np.where(track_ini<=Bad_points[i]))
#     check1=np.argmax(track_ini[track_ini<=Bad_points[i]])
#     #check2=np.amin(np.where(track_end>=Bad_points[i]))
#     #if (check1-check2) != 0:
#     #    print('i=',i,'check=',check1-check2)
#     Bad_tracks[check1]=1

# np.searchsorted(track_ini, Bad_points, side='right') returns the index of the first element in track_ini that is greater than or equal to Bad_points 
# The side='right' argument specifies that if the value is found, the index of the first occurrence should be returned. 

Bad_tracks[np.searchsorted(track_ini, Bad_points, side='right')-1] = True

np.savez('bad_tracks_' +run+ '.npz',Bad_tracks=Bad_tracks)

#%% TEST to verify that all the new method return the same result as the old one

if test:
    data=np.load('bad_tracks_' +run+ '.npz')
    Bad_tracks_new_method = data['Bad_tracks']
    data_test=np.load('bad_tracks_' + run + '_test.npz')
    Bad_tracks_old_method = data_test['Bad_tracks']
    # Convert both arrays to boolean
    Bad_tracks_old_method = Bad_tracks_old_method.astype(bool)
    # test if the two arrays are equal

    if np.array_equal(Bad_tracks_new_method,Bad_tracks_old_method)==False:
        print("ERROR: Both are not equal")
    else:
        print("OK : Both methods for Bad_tracks are equal")



#%% TEST to verify that all the saved samples have been estimated with more than 10 neighbors

if test:
    data = np.load('bad_tracks_' + run + '.npz')
    Bad_tracks = data['Bad_tracks']

    for i in range(dr_total_number_of_track-1):
        if not(Bad_tracks[i]):  # If the track is good
            if np.any(knn[track_ini[i]:track_ini[i+1]] < 10):  # Vérify if any sample in the track has less than 10 neighbors
                 print(f"ERROR: Track {i} has samples with less than 10 neighbors")


    print("OK, first filter done")


#%% Now we avoid tracks with less than 20 time samples 

# Array of lengths of tracks
#len_tracks=np.hstack((np.diff(track_ini)-1,len(knn)-track_ini[-1]))

# # Array of lengths of tracks for tracks with all the Drivas samples estimated with more than 10 neighbors
# len_of_tracks=((1-Bad_tracks)*len_tracks)  
# len_of_tracks=len_of_tracks.astype(int)

# # Final array with index of Good tracks
# Good_tracks=np.where(len_of_tracks>20)

# Bad_tracks is a mask of booleans, so we can use it directly to filter the tracks

#Good_tracks_id = np.where(~Bad_tracks & (len_tracks > 20))


# DR index of samples for good tracks are between Ini_good_tracks[i] and End_good_tracks[i] for i in len(Good_tracks)
# Ini_good_tracks=track_ini[Good_tracks]
# End_good_tracks=track_end[Good_tracks]


#Make INI and END good tracks integers
#Ini_good_tracks=Ini_good_tracks.astype(int)
#End_good_tracks=End_good_tracks.astype(int)

#Initialization of index_fit
# index_fit=-1
# index_with_errors=-1
# index_dri=-1
#siz=0


#%% TEST to verify that Index_from_initial_STB corresponds to the index transition from dr to tf
iSTB=np.array(dr['Index_from_initial_STB']).astype(int)-1 #(-1 for Python)

issue_with_index = np.ones(dr_total_number_of_track, dtype=bool)

nb_error=0
# Check if the number of sample in the track is equal to the number of samples in the STB
for i in range(dr_total_number_of_track-1):
    if track_end[i]-track_ini[i] != iSTB[track_end[i]]-iSTB[track_ini[i]]:
        print("ERROR: Track length in dr and STB do not match for track", i)
        print("Track length in dr: ", track_end[i]-track_ini[i])
        print("Track length in STB: ", iSTB[track_end[i]]-iSTB[track_ini[i]])
        issue_with_index[i] = False
        nb_error+=1
print("Number of errors: ", nb_error)
#%% Create mask for the filtered tracks

# Array of lengths of tracks
len_tracks=track_end-track_ini +1

# mask of good tracks
mask_dr = ~Bad_tracks & (len_tracks > 20) & issue_with_index
print("Ratio of tracks we keep: ",sum(mask_dr)/dr_total_number_of_track)

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

#%% Save file with the filtered tracks

# Save the filtered data to a new NetCDF file
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

print("filtered tracks saved in", output_path)
#%% Now we link indices from dr to indices from tf

# Read index from STB
iSTB=np.array(dr['Index_from_initial_STB']).astype(int)

# Ux,Ax index of samples for good tracks are dr['Index_from_initial_STB'] between Ini_good_tracks[i] and End_good_tracks[i]+1 for i in len(Good_tracks)
for i in range(len(Good_tracks[0])):
    #print(i,'  of  ',len(Good_tracks[0]))
    tmp=iSTB[Ini_good_tracks[i]:End_good_tracks[i]+1] # +1 here since the python command A:B goes from A until B-1
    tmp2=np.arange(Ini_good_tracks[i],End_good_tracks[i]+1)
    #siz+=len(tmp)
    if (np.sum(np.diff(tmp))-(len(tmp)-1))==0:
        index_fit=np.hstack((index_fit,tmp))
        index_dri=np.hstack((index_dri,tmp2))
    if test:
        if (np.sum(np.diff(tmp))-(len(tmp)-1))!=0:   # I don't understand this error ... It should not appear
            index_with_errors=np.hstack((index_with_errors,i)) # indices of Good_tracks with errors
            print("ERROR",i)
    
index_fit=index_fit[1::]-1      # -1 since iSTB indices are in Matlab notation. Our vector index_fit contains now indices starting at 0
index_dri=index_dri[1::]  # Here Ini_good_tracks is already in Python notation so we don't do -1
index_with_errors=index_with_errors[1::]  # Here God_tracks is already in Python notation so we don't do -1

np.savez('index_fit_run01.npz',index_fit=index_fit,index_dri=index_dri,index_with_errors=index_with_errors)

#%% TEST to verify that the saved indices correspond to the same positions for the fr and the tf files. 
if test:
    data=np.load('index_fit_run01.npz')
    index_fit=data['index_fit']
    index_dri=data['index_dri']
    index_with_errors=data['index_with_errors']

    drx=np.array(dr['x'])
    dry=np.array(dr['y'])
    drz=np.array(dr['z'])
    tfx=np.array(tf['x'])
    tfy=np.array(tf['y'])
    tfz=np.array(tf['z'])

    #for i in range(len(Good_tracks[0])):
    for i in range(len(Good_tracks[0])):
        #print(i)
        a=drz[Ini_good_tracks[i]:End_good_tracks[i]+1]
        b=2*tfz[iSTB[Ini_good_tracks[i]:End_good_tracks[i]+1]-1]
        c=2*tfz[index_fit[np.sum(len_of_tracks[Good_tracks[0][0:i]])+i:i+np.sum(len_of_tracks[Good_tracks[0][0:i]])+len_of_tracks[Good_tracks[0][i]]+1]]
        if np.sum((a-b))!=0:
            print("ERROR Z b",i)
        if np.sum((a-c))!=0:
            print("ERROR Z c",i)
        a=dry[Ini_good_tracks[i]:End_good_tracks[i]+1]
        b=2*tfy[iSTB[Ini_good_tracks[i]:End_good_tracks[i]+1]-1]
        c=2*tfy[index_fit[np.sum(len_of_tracks[Good_tracks[0][0:i]])+i:i+np.sum(len_of_tracks[Good_tracks[0][0:i]])+len_of_tracks[Good_tracks[0][i]]+1]]
        if np.sum((a-b))!=0:
            print("ERROR Y b",i)
        if np.sum((a-c))!=0:
            print("ERROR Y c",i)
        a=drx[Ini_good_tracks[i]:End_good_tracks[i]+1]
        b=2*tfx[iSTB[Ini_good_tracks[i]:End_good_tracks[i]+1]-1]
        c=2*tfx[index_fit[np.sum(len_of_tracks[Good_tracks[0][0:i]])+i:i+np.sum(len_of_tracks[Good_tracks[0][0:i]])+len_of_tracks[Good_tracks[0][i]]+1]]
        if np.sum((a-b))!=0:
            print("ERROR X b",i)
        if np.sum((a-c))!=0:
            print("ERROR X c",i)

#%% TEST on positions

# ii=Good_tracks[0][index_with_errors[1]]

# iSTB=(np.array(dr['Index_from_initial_STB'])-1).astype(int)

# np.arange(track_ini[ii],track_end[ii])

# iSTB[track_ini[ii]:track_end[ii]]

# xdr=drx[track_ini[ii]:track_end[ii]+1]
# xtf=2*tfx[iSTB[track_ini[ii]]:iSTB[track_end[ii]]+1]

# ydr=dry[track_ini[ii]:track_end[ii]+1]
# ytf=2*tfy[iSTB[track_ini[ii]]:iSTB[track_end[ii]]+1]

# zdr=drz[track_ini[ii]:track_end[ii]+1]
# ztf=2*tfz[iSTB[track_ini[ii]]:iSTB[track_end[ii]]+1]

# # xdr=drx[track_ini[ii]:track_end[ii]+10]
# # xtf=2*tfx[iSTB[track_ini[ii]]:iSTB[track_end[ii]]+10]

# # ydr=dry[track_ini[ii]:track_end[ii]+10]
# # ytf=2*tfy[iSTB[track_ini[ii]]:iSTB[track_end[ii]]+10]

# # zdr=drz[track_ini[ii]:track_end[ii]+10]
# # ztf=2*tfz[iSTB[track_ini[ii]]:iSTB[track_end[ii]]+10]


#%%
'''TrackFit.nc file structure:'''
'''
{'x': <class 'netCDF4._netCDF4.Variable'>
 float32 x(dim_points)
     unit: meters
 current shape = (232928519,)

 'y': <class 'netCDF4._netCDF4.Variable'>
 float32 y(dim_points)
     unit: meters
 current shape = (232928519,)

 'z': <class 'netCDF4._netCDF4.Variable'>
 float32 z(dim_points)
     unit: meters
 current shape = (232928519,)

 'u': <class 'netCDF4._netCDF4.Variable'>
 float32 u(dim_points)
     unit: meters/second
 current shape = (232928519,)

 'v': <class 'netCDF4._netCDF4.Variable'>
 float32 v(dim_points)
     unit: meters/second
 current shape = (232928519,)

 'w': <class 'netCDF4._netCDF4.Variable'>
 float32 w(dim_points)
     unit: meters/second
 current shape = (232928519,)

  'Image_Acquisition_Frequency': <class 'netCDF4._netCDF4.Variable'>
 float32 Image_Acquisition_Frequency(recording_information)
     unit: [107.  72. 122.]
 current shape = (1,)

 'time_track_begin': <class 'netCDF4._netCDF4.Variable'>
 int32 time_track_begin(total_number_of_track)
 current shape = (12491729,)

 'track_length': <class 'netCDF4._netCDF4.Variable'>
 int32 track_length(total_number_of_track)
 current shape = (12491729,)

 'track_start_index': <class 'netCDF4._netCDF4.Variable'>
 int32 track_start_index(total_number_of_track)
 current shape = (12491729,)
  '''
