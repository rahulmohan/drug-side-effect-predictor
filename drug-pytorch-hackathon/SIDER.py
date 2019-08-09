import torch
import numpy as np
import pandas as pd
from torch.utils import data
from LogisticRegression import *

class SIDER(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, inputs_file="sider_latent.npy", outputs_file="SIDER_PTs.csv", IDs_file="sider_idxs_to_keep.npy"):
        'Initialization'
        latent = np.load(inputs_file)
        idxs_to_keep = np.load(IDs_file)
        sider_csv = pd.read_csv(outputs_file)
        sider = sider_csv.values
        sider_labels = sider[idxs_to_keep, 1:].astype('int')
        label_names = sider_csv.columns.values.tolist()  # side effect names
        self.effects_dict = {i - 1: label_names[i] for i in range(1, len(label_names))}
        self.labels = torch.from_numpy(sider_labels)
        self.inputs = torch.from_numpy(latent)
        self.IDs = torch.from_numpy(idxs_to_keep)
        self.IDs_dict = sider[:,0]
        self.input_len = self.inputs.shape[1]
        self.label_classes = self.labels.shape[1]
        #print("labels.size", self.labels.shape[1])
#        self.len_latent =

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.IDs_dict[self.IDs[index]], self.IDs[index]
        # Load data and get label
        X = self.inputs[index]
        y = self.labels[index]
        return X, y, ID

