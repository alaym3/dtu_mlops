import torch
import numpy as np
import glob


class Dataset():
    def __init__(self,data,target):
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data[idx]
        y = self.target[idx]
    
        return X,y


def mnist():
    # set file paths
    train_files = glob.glob(r'/Users/alaina/Desktop/classes/2023Jan/dtu_mlops/data/corruptmnist/train*.npz')
    test_files = glob.glob(r'/Users/alaina/Desktop/classes/2023Jan/dtu_mlops/data/corruptmnist/test*.npz')

    # initialize X and y arrays for training
    train_images = []
    train_labels = []
    for file in train_files:
        data = np.load(file)
        train_images.append(data['images'])
        train_labels.append(data['labels'])
    train_images = np.concatenate((train_images),axis=0)
    train_labels = np.concatenate((train_labels),axis=0)

    # initialize X and y arrays for testing
    test_images = []
    test_labels = []
    for file in test_files:
        data = np.load(file)
        test_images.append(data['images'])
        test_labels.append(data['labels'])
    test_images = np.concatenate((test_images),axis=0)
    test_labels = np.concatenate((test_labels),axis=0)

    # convert training data to appropriate format
    train_images = torch.from_numpy(train_images).float()
    train_labels = torch.from_numpy(train_labels).long()

    # convert test data to appropriate format
    test_images = torch.from_numpy(test_images).float()
    test_labels = torch.from_numpy(test_labels).long()

    # return a tuple for train and test respectively using the Dataset class
    train = Dataset(train_images,train_labels)
    test = Dataset(test_images,test_labels) 
    return train, test