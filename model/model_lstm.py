import torch.nn as nn
import torch

train_on_gpu = torch.cuda.is_available()


class LSTM(nn.Module):
    def __init__(self, input_size=12, hidden_dim=100, n_layers=5, output_size=7, drop_prob=0.5, lr=0.001, batch_first=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.lr = lr

        # define the LSTM
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=n_layers, dropout=drop_prob, batch_first=batch_first)

        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # define the final, fully-connected output layer
        self.linear = nn.Linear(5000, output_size)

    def forward(self, input_seq, hidden):
        lstm_out, hidden = self.lstm(input_seq, None)
        out = lstm_out[:, :, -1]
        out = self.dropout(out)

        # fully-connected layer
        out = self.linear(out)

        # return the final output and the hidden state
        return out, hidden
