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
from datetime import datetime

time_ub = 60
num_moments = 5

class my_Dataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, index):
        # print(self.data_paths[index])
        x, y = pkl.load(open(self.data_paths[index] , 'rb'))
        if self.data_paths[index].split('/')[-1].startswith('ich'):
            inputs = torch.from_numpy(x[:,:time_ub,:])
            x = inputs #torch.cat((inputs[:, :,:5], inputs[:, :,10:15], inputs[:,:,20:]), 2)
        else:
            inputs = torch.from_numpy(x[:,:time_ub,:21+15])
            x = torch.cat((inputs[:, :,:5], inputs[:, :,10:15], inputs[:,:,20:]), 2)
        y = torch.from_numpy(y[:, :, :])

        return x, y

    # def __getitem__(self, index):
    #     # print(self.data_paths[index])
    #     x, y = pkl.load(open(self.data_paths[index], 'rb'))
    #     inputs = torch.from_numpy(x[:, :time_ub, :21 + 15])
    #     x = torch.cat((inputs[:, :, :num_moments], inputs[:, :, 10:10+num_moments], inputs[:, :, 20:]), 2)
    #     y = torch.from_numpy(y[:, :, :])
    #     # y = (y[:,:time_ub,:]*torch.arange(71)).sum(axis = 2)
    #     # y = y.reshape((16, time_ub,1))
    #
    #     # x= torch.cat((torch.exp(inputs[:, :,:4]), inputs[:, :,10:14], inputs[:,:,20:]), 2)
    #
    #     return x, y


class RNN1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN1, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # .to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # .to(device)

        out, _ = self.lstm(x.float(), (h0.float(), c0.float()))


        out = self.fc(out)

        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x.float(), (h0.float(), c0.float()))


        out = self.fc(out)

        return out

def loss_queue(soft, labels):
    return ((torch.abs(soft[:, :, :] - labels[:, :, :])).sum(axis = [2]) + torch.max(torch.abs(soft[:, :, :] - labels[:,:,:]), axis = 2)[0]).mean()

    # return ((torch.abs(soft[:,:,:] - labels[:,:,:])).sum(axis = [2]) + torch.max(torch.abs(soft[:,:,:] - labels[:,:,:]), axis = 2)[0]).mean()


def valid_score(soft, labels):
    return ((torch.abs(soft[:, :,:] - labels[:,:,:])).sum(axis = [2]) ).mean()



# loss = nn.CrossEntropyLoss()

def main(args):

    loss_path =   '/scratch/eliransc/RNN_loss_new'
    models_path = '/scratch/eliransc/RNN_models_new'

    path = '/scratch/eliransc/new_gt_g_1_batches1/' # Ichilov_gt_g_1_folders/'
    file_list = os.listdir(path)
    data_paths = [os.path.join(path, name) for name in file_list]

    valid_path_ = '/scratch/eliransc/new_gt_g_1_batches1_valid/'  # Ichilov_gt_g_1_folders
    valid_path = os.listdir(valid_path_)

    valid_path = [os.path.join(valid_path_, name) for name in valid_path]

    # create dataset
    dataset = my_Dataset(data_paths)
    dataset_valid = my_Dataset(valid_path)

    num_epochs = 15
    lr_change = np.random.choice([1.025, 1.05, 1.1, 1.25, 1.3], p=[0.2, 0.2, 0.2, 0.2, 0.2])
    batch_size = int(np.random.choice([1, 2, 4, 8], p=[0.25, 0.35, 0.25, 0.15]))
    learning_rate = np.random.choice([0.001, 0.005, 0.0005, 0.0002, 0.0001], p=[0.2, 0.2, 0.2, 0.2, 0.2])

    # input_size = 36
    hidden_size = int(np.random.choice([32, 64, 128], p=[0.3, 0.4, 0.3]))
    num_layers = np.random.randint(2, 6)


    input_size = 16+2*num_moments
    sequence_length = time_ub
    output_size = 51

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    valid_loader = DataLoader(dataset=dataset_valid,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0)

    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model1 = RNN1(input_size, hidden_size, num_layers, output_size)  # .to(device)
    m = nn.Softmax(dim=2)
    # Dummy Training loop

    n_iterations = 5
    n_total_steps = 20
    loss_list = []
    setting_string = 'batch_size_' + str(batch_size * 32) + '_num_layers_' + str(num_layers) + '_num_epochs_' + str(
        num_epochs) + '_learning_rate_' + str(learning_rate) + '_hidden_size_' + str(hidden_size) + '_lr_change_' + str(
        lr_change) + '_nummoms_' + str(num_moments)
    print(setting_string)

    valid_loss_list = []
    now = datetime.now()

    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])

    for epoch in range(num_epochs):

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=(1 / 10 ** 5))

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
            labels = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2], labels.shape[3])

            inputs = inputs.float()
            labels = labels.float()

            if ((inputs.sum()).isfinite()) & (inputs[inputs < -1111110].shape[0] == 0):

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

                # print('i is', i)
                if (i + 1) % 50 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')


                    torch.save(model.state_dict(), os.path.join(models_path, 'pytorch_gt_gt_1_true_moms_new_data_withtest2_' + setting_string + '_' + str(
                                   current_time) + '.pkl'))
            else:
                print(i)

        learning_rate = learning_rate ** lr_change
        # print(inputs[inputs<0].shape)

        torch.save(model.state_dict(), os.path.join(models_path, 'pytorch_gt_gt_1_true_moms_new_data_withtest2_' + setting_string + '_' + str(
                                   current_time) + '.pkl'))

        model1.load_state_dict(
            torch.load(os.path.join(models_path, 'pytorch_gt_gt_1_true_moms_new_data_withtest2_' + setting_string + '_' + str(
                                   current_time) + '.pkl'), map_location=torch.device('cpu')))
        totloss = []

        for i, (inputs, labels) in tqdm(enumerate(valid_loader)):
            if i < 1000:

                # print(inputs.shape)
                inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
                labels = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2], labels.shape[3])

                inputs = inputs.float()
                labels = labels.float()

                if ((inputs.sum()).isfinite()) & (inputs[inputs < -1111110].shape[0] == 0):
                    inputs = inputs.reshape(-1, sequence_length, input_size)  # .to(device)

                    # labels = labels.to(device)

                    # Forward pass
                    outputs = model1(inputs)
                    soft = m(outputs)
                    totloss.append(valid_score(soft, labels))
                    # print('valid', torch.tensor(totloss).mean())

        valid_loss_list.append(torch.tensor(totloss).mean())
        pkl.dump((loss_list, valid_loss_list),
                 open(os.path.join(loss_path, 'pytorch_gt_gt_1_true_moms_new_data_withtest2_' + setting_string + '_' + str(
                                   current_time) + '.pkl'), 'wb'))



def parse_arguments(argv):

    parser = argparse.ArgumentParser()


    parser.add_argument('--dump_path', type=str, help='path to pkl folder', default= r'C:\Users\user\workspace\data\gg1_inverse_pkls' ) # '/scratch/eliransc/gg1_inverse_pkls'
    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
