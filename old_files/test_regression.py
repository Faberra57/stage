#%%
import toolkit as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


working_directory="/Volumes/T7 Shield/ALEAS/"
tk.set_working_directory(working_directory)

print("loading data...")
run = 1
p = 3
data_folder = "Split_Tracks"
df = tk.open_data(data_folder, run,p)
df.head()
db = df.to_numpy()


# Parameters
k = 5

# Train and Test Split
print("Splitting data into train and test sets...")
db_train = db[:-int(len(db)/10)]
db_test = db[-int(len(db)/10):]

#%%
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import pinv

weighting = True

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
weighted_covariance = np.zeros_like(vals)
#%%
print("Calculating coefficients...")
x = db_train[idx, :-1]
x_test = db_test[:, :-1]
y = (weights * db_train[idx, -1])[:, :, np.newaxis]

x = np.pad(x, [(0, 0), (0, 0), (1, 0)], mode='constant', constant_values=1) 
x_test = np.pad(x_test, [(0, 0), (1, 0)], mode='constant', constant_values=1)
coef = pinv(np.transpose(x, axes=[0, 2, 1]) @ (weights[:, :, np.newaxis] * x)) @ np.transpose(x, axes=[0, 2, 1]) @ y
coef = np.squeeze(coef)
#%%

vals = np.sum(coef * x_test, axis=1)


#residuals = db_train[idx, -1] - np.sum(coef * np.transpose(x, (0, 2, 1)), axis=1)
#weighted_covariance = (np.sum(weights * residuals**2, axis=1) / np.sum(weights, axis=1))

# if normalize:
#     vals[(p - 1) * stride + 1:] += puplet_means[:-1]

print("Comparing results...")
from sklearn.metrics import mean_squared_error
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(db_test[:, -1], vals))
print(f"RMSE: {rmse:.2f}")


# %%
