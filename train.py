import numpy as np
from model.model_cnn_1d import *
from utils.utils_cnn import ECGDatasets, ecg_dataloader
from sklearn.metrics import accuracy_score
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16


def train(net, train_data_loader, val_data_loader, epochs=20, batch_size=batch_size, lr=0.1, print_every=3):
    ''' Training a network
        Arguments
        ---------
        :param print_every: Number of steps for printing training and validation loss
        :param net: LSTM network
        :param clip: gradient clipping
        :param lr: learning rate
        :param batch_size: Number of mini-sequences per mini-batch, aka batch size
        :param epochs: Number of epochs to train
        :param val_data_loader: ecg data to train the network
        :param train_data_loader: ecg data to train the network

    '''
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # 定义损失函數，这里我们使用binary cross entropy loss

    if train_on_gpu:
        net.to(device)

    counter = 0
    for e in range(epochs):
        for x, y in tqdm(train_data_loader, desc='training'):
            counter += 1
            x = torch.unsqueeze(x, 2)
            inputs, targets = x.to(device), y.to(device)

            if train_on_gpu:
                inputs, targets = inputs.to(device), targets.to(device)

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            inputs = torch.squeeze(inputs)
            output, h = net(inputs, None)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.to(dtype=torch.long))
            loss.backward()
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_losses = []
                acc_List = []
                net.eval()
                with torch.no_grad():
                    for x_val, y_val in val_data_loader:
                        if train_on_gpu:
                            x_val, y_val = x_val.to(device), y_val.to(device)

                        output_val, _ = net(x_val, None)
                        val_loss = criterion(output_val, y_val.to(dtype=torch.long))
                        pred_label = output_val.argmax(1)
                        acc = accuracy_score(y_val.cpu().numpy(), pred_label.cpu().numpy())
                        val_losses.append(val_loss.item())
                        acc_List.append(acc)

                net.train()  # reset to train mode

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)),
                      "acc: {:.4f}".format(np.mean(acc_List)))


train_data = ECGDatasets('data/index_ecg_train.txt')
val_data = ECGDatasets('data/index_ecg_val.txt')

train_loader = ecg_dataloader(train_data, batch_size)
val_loader = ecg_dataloader(val_data, batch_size)


hidden_dim = 64
n_layers = 3
net = LSTM(12, hidden_dim=hidden_dim, n_layers=n_layers)


if __name__ == '__main__':
    train(net, train_loader, val_loader, epochs=10, batch_size=batch_size, lr=0.001, print_every=2)
