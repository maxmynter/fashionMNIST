import pandas as pd
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

class FashionData(Dataset):
    def __init__(self, dataPath, nrows = None):
        super(FashionData, self).__init__()
        if nrows:
            data  = pd.read_csv(dataPath, nrows=nrows) 
        else:
            data  = pd.read_csv(dataPath) 
        self.labels =  torch.tensor(np.array(data['label']))
        self.images = torch.tensor(np.array(data[data.columns[1:]]).reshape(len(data),1,28,28).astype('float32'))

    def __getitem__(self, index):
        return self.labels[index], self.images[index]

    def __len__(self):
        return len(self.images)


class FashionBaselineModel():
    def __init__(self, labels, data):
        self.unique_labels = np.unique(labels)
        self.labels = labels
        averages = {}
        for label in self.unique_labels:
            select_of_label = data[labels==label, :,:]
            averages[label] = torch.sum(select_of_label, dim=0)/select_of_label.shape[0]
        self.averages = averages
        performance =self.score_on_data_set(labels, data)
        print("Training set Accuracy: " , performance[0])
        for lab in performance[1].keys():
            print(f"Performance for target label {lab}", performance[1][lab])
    
    def score(self, target):
        distances = {}
        for label in self.unique_labels:
            average = self.averages[label]
            distance =torch.sum(torch.abs(average - target)) 
            distances[label] = distance
        return min(distances, key=distances.get)
    
    def score_on_data_set(self, labels, data):
        labels = np.array(labels)
        n_correct = 0 
        n_correct_per_label = {lab: 0 for lab in np.unique(labels)}
        for i in range(len(labels)):
            target_label = labels[i]
            is_correct = self.score(data[i,:,:]) == target_label
            if is_correct:
                n_correct += 1
                n_correct_per_label[target_label] += 1
        n_correct_per_label = {lab: n_correct_per_label[lab]/np.count_nonzero(labels == lab) for lab in n_correct_per_label.keys()}
        return n_correct / len(labels), n_correct_per_label
