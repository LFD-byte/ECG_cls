import torch.nn as nn
import torch
from copy import deepcopy

train_on_gpu = torch.cuda.is_available()


class LSTM(nn.Module):
    def __init__(self, input_size=12, hidden_dim=100, n_layers=5, output_size=7, drop_prob=0.5, lr=0.001, batch_first=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 32, 3, 1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.Fc = nn.Sequential(
            nn.Linear(39968, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7)
        )
        self.MaxPool = nn.MaxPool1d(3, stride=2)
        self.AvePool = nn.AvgPool1d(3, stride=2)
        self.Linear = nn.Linear(39968, 7)

    def forward(self, input_seq, hidden):
        # print(input_seq.shape)
        x = self.conv1(input_seq)
        # print(x.shape)
        x = self.MaxPool(x)
        x_ = x

        x = self.conv2(x)
        x = self.conv2(x)
        x = x + x_

        x = self.AvePool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        out = self.Fc(x)

        return out, None
