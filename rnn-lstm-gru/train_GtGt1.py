import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle as pkl
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import sys


class my_Dataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        # print(self.data_paths[index])
        x, y = pkl.load(open(self.data_paths[index], 'rb'))
        x = torch.from_numpy(x[:, :40, :21 + 15])
        y = torch.from_numpy(y[:, :40, :])

        return x, y


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        # or:
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 28, 28), h0: (2, n, 128)

        # Forward propagate RNN
        # out, _ = self.gru(x.float(), h0.float())
        # or:
        out, _ = self.lstm(x.float(), (h0.float(), c0.float()))

        # out: (n, 128)

        out = self.fc(out)

        # out: (n, 10)
        return out


def loss_queue(soft, labels):
    return ((torch.abs(soft - labels)).sum(axis=[2]) + torch.max(torch.abs(soft - labels), axis=2)[0]).mean()


# loss = nn.CrossEntropyLoss()

def main(args):


    path = '/scratch/eliransc/GtGt1_from_narval/GtGt1_to_beluga'
    file_list = os.listdir(path)
    data_paths = [os.path.join(path, name) for name in file_list]

    # create dataset
    dataset = my_Dataset(data_paths)
    dataset_valid = my_Dataset(data_paths[:20])

    num_epochs = np.random.randint(5, 10)
    batch_size = int(np.random.choice([1, 2, 4, 16, 32], p=[0.2, 0.2, 0.2, 0.2, 0.2]))
    learning_rate = np.random.choice([0.0001, 0.005, 0.0005, 0.0002, 0.00001], p=[0.2, 0.2, 0.2, 0.2, 0.2])

    input_size = 36
    sequence_length = 40
    hidden_size = int(np.random.choice([32, 64, 128], p=[0.3, 0.4, 0.3]))
    num_layers = np.random.randint(2, 10)
    output_size = 71

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

    valid_loader = DataLoader(dataset=dataset_valid,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)



    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)

    m = nn.Softmax(dim=2)
    # Dummy Training loop


    n_total_steps = 10
    loss_list = []
    setting_string = 'batch_size_' + str(batch_size * 16) + '_num_layers_' + str(num_layers) + '_num_epochs_' + str(
        num_epochs) + '_learning_rate_' + str(learning_rate) + '_hidden_size_' + str(hidden_size)

    # print(total_samples, n_iterations)
    now = datetime.now()

    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])

    for epoch in range(num_epochs):


        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        learning_rate = learning_rate*0.9
        print(learning_rate)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
            labels = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2], labels.shape[3])

            inputs = inputs.float()
            labels = labels.float()

            if (inputs.sum()).isfinite():

                inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(inputs)
                soft = m(outputs)
                loss = loss_queue(soft, labels)  # criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

        torch.save(model.state_dict(),
                   '/scratch/eliransc/RNN_models/pytorch_gt_gt_1_true_moms_new_data_' + setting_string + '_' + str(
                       current_time) + '.pkl')

        pkl.dump(loss_list,
                 open('/scratch/eliransc/RNN_models/' + 'loss_' + setting_string + '_' + str(
                     current_time) + '.pkl', 'wb'))


def parse_arguments(argv):

    parser = argparse.ArgumentParser()


    parser.add_argument('--dump_path', type=str, help='path to pkl folder', default= r'C:\Users\user\workspace\data\gg1_inverse_pkls' ) # '/scratch/eliransc/gg1_inverse_pkls'
    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
