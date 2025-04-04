import Toolkit as tk
if __name__ == "__main__":

    # Parameters
    working_directory = "/Volumes/T7 Shield/ALEAS/"
    nb_runs = 11 # Number of runs
    nmin = 10 # Minimum number of neighbors for calculating epsilon
    len_min = 20 # Minimum length of tracks
    p = 3 # Number of points to consider for the split
    filtered_data_folder = "Filtered_Tracks" # Folder to save filtered tracks
    split_data_folder = "Split_Tracks" # Folder to save split tracks

    # Action to perform
    set_working_directory = True
    filtering_raw_data = True
    spliting_raw_data = True

    if set_working_directory: tk.set_working_directory(working_directory)
    if filtering_raw_data:
        if tk.check_folder(filtered_data_folder):
            print ("Filtered data folder already exists. Skipping filtering.")
        else:
            print ("Folder for filtered data does not exist. Creating folder...")
            print ("Starting filtering...")
            for i in range(1, 11):
                tk.filter_tracks(i,filtered_data_folder, nmin=10, len_min=20)
            print("Filtering completed.")
    if spliting_raw_data:
        if tk.check_folder(split_data_folder):
            print ("Split data folder already exists. Skipping splitting.")
        else:
            print ("Folder for split data does not exist. Creating folder...")
            print ("Starting splitting...")
            for i in range(1, 11):
                tk.split_data(i, p, filtered_data_folder, split_data_folder)
            print("Splitting completed.")