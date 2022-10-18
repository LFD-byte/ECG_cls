# coding=utf-8

import scipy.io as sio
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np


class ECGDatasets(Dataset):
    def __init__(self, txt_path):
        self.ecgs = []
        with open(txt_path, 'r') as f:
            ecgs = f.readlines()
        for ecg in ecgs:
            self.ecgs.append(ecg.split(' '))

    def __getitem__(self, item):
        ep, label = self.ecgs[item]
        ecg = np.asarray(sio.loadmat(ep)['val'], dtype=np.float32)
        label = np.asarray(int(label.replace('\n', '')), dtype=np.float32)
        ecg = torch.from_numpy(ecg)
        label = torch.from_numpy(label)

        return ecg, label

    def __len__(self):
        return len(self.ecgs)


def ecg_dataloader(data, batch_size):
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=1)
    return data_loader
# drop_last=True

