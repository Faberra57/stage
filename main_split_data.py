import Toolkit as tk

if __name__ == "__main__":
   working_directory="/Volumes/T7 Shield/ALEAS/"
   processed_data_folder="Split_Tracks"
   run_list=["run01","run02","run03","run04","run05"] 
   p=3
   tk.split_data(working_directory,processed_data_folder,run_list,p)