import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Colordataset(Dataset):

    def __init__(self, file_data, file_label):
        self.data = np.loadtxt(file_data)
        self.label = np.loadtxt(file_label)


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx,:]
        label = self.label[idx,:] 
        sample = {'data': data, 'label': label}

        return sample

if __name__ == '__main__':

    colordataset = Colordataset(file_data = 'xyY.txt', file_label = 'Lab.txt')
    dataloader = DataLoader(colordataset, batch_size=4,
                           shuffle=True, num_workers=2)
    for i_batch, sample_batched  in enumerate(dataloader):
        print(i_batch)
        print(sample_batched['data'])
        print(sample_batched['label'])

