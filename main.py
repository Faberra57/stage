import toolkit as tk
if __name__ == "__main__":

    # Parameters
    working_directory = "/Volumes/T7 Shield/ALEAS/"
    nb_runs = 11 # Number of runs
    nmin = 10 # Minimum number of neighbors for calculating epsilon
    len_min = 20 # Minimum length of tracks
    p = 3 # Number of points to consider for the split
    filtered_data_folder = "Filtered_Tracks" # Folder to save filtered tracks
    split_data_folder = "Split_Tracks" # Folder to save split tracks

    training_ratio = 0.8 # Ratio of training data
    k = 5 # Number of neighbors for prediction

    # Action to perform
    set_working_directory = True # Set working directory
    filtering_raw_data = False # Filtering raw data
    spliting_raw_data = False # Splitting raw data
    predicting_with_param = True # Predicting with parameters above
    predicting_with_testing_param = False # Predicting with testing parameters

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

    if predicting_with_param:
        nb_run = 1
        print (f"Starting prediction with parameters p={p} and k={k}...")
        db = tk.load_database(nb_run,p,split_data_folder,filtered_data_folder)
        y_predicted , y_test = tk.prediction_neighbors(db,training_ratio,k)
        print("Prediction completed.")
        result = tk.stats_results(y_predicted,y_test)
        print ("Results:")
        print ("MSE: ", result[0])
        print ("RMSE: ", result[1])
        print ("Correlation: ", result[2])
        print ("MAE: ", result[3])
        print ("R^2: ", result[4])

    if predicting_with_testing_param:
        results_list = []
        db = tk.load_database(working_directory)
        p_test = [1,3,5]
        k_test = [5,20,50]
        nb_run = 1 # Number of run
        print ("Starting prediction...")
        for p in p_test:
            for k in k_test:
                print (f"Starting prediction for p={p} and k={k}...")
                db = tk.load_database(split_data_folder,nb_run,p)
                y_predicted , y_test = tk.prediction_neighbors(db,training_ratio,k)
                result = tk.stats_results(y_predicted,y_test)
                print (f"Results for p={p} and k={k}:")
                print ("MSE: ", result[0])
                print ("RMSE: ", result[1])
                print ("Correlation: ", result[2])
                print ("MAE: ", result[3])
                print ("R^2: ", result[4])
                results_list.append((p,k,result))
                print ("-------------------------------------")
        # Save results to a file
        with open("results.txt", "a") as f:
            for p, k, result in results_list:
                f.write(f"p={p}, k={k}, MSE={result[0]}, RMSE={result[1]}, Correlation={result[2]}, MAE={result[3]}, R^2={result[4]}\n")
        print("Results saved to results.txt")
                

