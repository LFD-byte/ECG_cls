import torch.nn as nn
import torch


train_on_gpu = torch.cuda.is_available()

class LSTM(nn.Module):
    def __init__(self, input_size=12, hidden_dim=100, n_layers=5, output_size=7, drop_prob=0.5, lr=0.001, batch_first=True):
        super().__init__()
        self.CNN = nn.Conv2d(1, 32, 5)
        self.BN = nn.BatchNorm2d(32)
        self.Relu = nn.ReLU()
        self.Pool = nn.MaxPool2d(2, 2)
        self.Linear1 = nn.Linear(2498, 1)
        self.Linear2 = nn.Linear(4, 1)
        self.Linear3 = nn.Linear(32, 7)

    def forward(self, input_seq, hidden):
        x = torch.unsqueeze(input_seq, 1)
        x = self.CNN(x)
        # print(x.shape, 'CNN')
        x = self.BN(x)
        # print(x.shape, 'BN')
        x = self.Relu(x)
        x = self.Pool(x)
        # print(x.shape, 'Pool')

        x = torch.squeeze(x)
        # print(x.shape, 'squeeze')
        x = self.Linear1(x)
        # print(x.shape, 'Linear1')
        x = torch.squeeze(x)
        # print(x.shape, 'Squeeze')
        x= self.Linear2(x)
        # print(x.shape, 'Linear2')
        x = torch.squeeze(x)
        # print(x.shape, 'Squeeze')
        out = self.Linear3(x)
        # print(out.shape, 'Linear3')

        return out, None
