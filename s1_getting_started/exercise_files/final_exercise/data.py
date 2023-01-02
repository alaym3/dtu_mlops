import torch
import numpy as np
import os
import glob


def mnist():
   # initialize data arrays
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # search all files inside a specific folder
    # *.npz means file name only with npz extension
    dir_path = r'/Users/alaina/Desktop/classes/2023Jan/dtu_mlops/data/corruptmnist/*.npz'
    for file in glob.glob(dir_path, recursive=True):
        # save the data and labels for each of the train and test files
        # save training first
        if 'train' in file:
            X_train.append(np.load(str(file))['images'])
            y_train.append(np.load(str(file))['labels'])
        # save test second
        elif 'test' in file:
            X_test.append(np.load(str(file))['images'])
            y_test.append(np.load(str(file))['labels'])
    # return train, test
    return (X_train, y_train) , (X_test, y_test)
