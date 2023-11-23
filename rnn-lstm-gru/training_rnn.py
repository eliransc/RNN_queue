import torch.nn as nn
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
from datetime import datetime
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = nn.Softmax(dim=2)
num_moments = np.random.randint(1,11)
time_ub = 60


def valid_loss(model, valid_loader,sequence_length, input_size):

    torch.cuda.empty_cache()

    totloss = []

    for i, (inputs, labels) in enumerate(valid_loader):
        if i < 1000:

            inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
            labels = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2], labels.shape[3])

            inputs = inputs.float()
            labels = labels.float()

            if ((inputs.sum()).isfinite()) & (inputs[inputs < -1111110].shape[0] == 0):
                inputs = inputs.reshape(-1, sequence_length, input_size)  # .to(device)

                # labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                # soft = m(outputs)
                totloss.append(valid_score(outputs, labels))


        else:
            break

    return torch.tensor(totloss).mean()


def check_loss_increasing(loss_list, n_last_steps=10, failure_rate=0.45):
    counter = 0
    curr_len = len(loss_list)
    if curr_len < n_last_steps:
        n_last_steps = curr_len

    inds_arr = np.linspace(n_last_steps - 1, 1, n_last_steps - 1).astype(int)
    for ind in inds_arr:
        if loss_list[-ind] > loss_list[-ind - 1]:
            counter += 1

    print(counter, n_last_steps)
    if counter / n_last_steps > failure_rate:
        return True

    else:
        return False


def loss_queue(soft, labels):
    return ((torch.abs(soft[:, :, :] - labels[:, :, :])).sum(axis=[2]) +
            torch.max(torch.abs(soft[:, :, :] - labels[:, :, :]), axis=2)[0]).mean()


def SAE_batch(soft, labels):
    return ((torch.abs(soft[:, :, :] - labels[:, :, :])).sum(axis=[2])).mean()


def loss_queue1(soft, labels):
    return (3 * (torch.abs(soft[:, :3, :] - labels[:, :3, :])).sum(axis=[2])).mean() + (
                (torch.abs(soft[:, :, :] - labels[:, :, :])).sum(axis=[2]) +
                torch.max(torch.abs(soft[:, :, :] - labels[:, :, :]), axis=2)[0]).mean()


def valid_score(soft, labels):
    return ((torch.abs(soft[:, 1:, :] - labels[:, 1:, :])).sum(axis=[2])).mean()


def valid_score_all_vals(soft, labels):
    return (torch.abs(soft[:, 1:, :] - labels[:, 1:, :])).sum(axis=[2])


def loss_func(output, target):
    # print(output.shape, target.shape)
    return (((output[:, :, :] - target[:, :, :]) ** 2) ** 0.5).mean()


class my_Dataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    # def __getitem__(self, index):
    #     x, y = pkl.load(open(self.data_paths[index] , 'rb'))
    #     if self.data_paths[index].split('/')[-1].startswith('ich'):
    #         inputs = torch.from_numpy(x[:,:time_ub,:])
    #         x = inputs #torch.cat((inputs[:, :,:5], inputs[:, :,10:15], inputs[:,:,20:]), 2)
    #     else:
    #         inputs = torch.from_numpy(x[:,:time_ub,:21+30])
    #         x = torch.cat((inputs[:, :,:5], inputs[:, :,10:15], inputs[:,:,21:]), 2)
    #     y = torch.from_numpy(y[:,:,:])
    #     return (x, y)

    def __getitem__(self, index):

        x, y = pkl.load(open(self.data_paths[index], 'rb'))

        inputs = torch.from_numpy(x[:, :time_ub, :21 + 30])
        x = torch.cat((inputs[:, :, :num_moments], inputs[:, :, 10:10 + num_moments], inputs[:, :, 21:]), 2)
        y = torch.from_numpy(y[:, :, :])

        return (x, y)


def SAE_valid(valid, sequence_length, input_size, model):
    SAE_vals = []

    for i, (inputs, labels) in enumerate(valid):
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
            loss = SAE_batch(soft, labels)
            SAE_vals.append(loss)

    return torch.tensor(SAE_vals).mean()


class RNN1(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN1, self).__init__()
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # .to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # .to(device)
        # x: (n, 28, 28), h0: (2, n, 128)

        # Forward propagate RNN
        # out, _ = self.gru(x.float(), h0.float())
        # or:
        out, _ = self.lstm(x.float(), (h0.float(), c0.float()))

        # out: (n, 128)

        out = self.fc(out)

        # out: (n, 10)
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
        # or:
        out, _ = self.lstm(x.float(), (h0.float(), c0.float()))
        out = self.fc(out)

        return out

def main(args):



    # 0: 'LN025', 1: 'LN4', 2: 'G4', 3: 'G025', 4: 'M'

    path = '/scratch/eliransc/rnn_data/training_trans_moments_fixed'  # '/scratch/eliransc/new_gt_g_1_trans4/' #  Ichilov_gt_g_1_folders
    file_list = os.listdir(path)
    data_paths = [os.path.join(path, name) for name in file_list]

    valid_path_ = '/scratch/eliransc/rnn_data/test_set_1_a'  # '/scratch/eliransc/rnn_data/gt_gt_1_batches_G4_experiment'
    valid_path = os.listdir(valid_path_)

    valid_path = [os.path.join(valid_path_, name) for name in valid_path]

    valid_path_1 = '/scratch/eliransc/rnn_data/testset1_batches/5.0'  # '/scratch/eliransc/rnn_data/gt_gt_1_batches_G4_experiment'
    valid_path1 = os.listdir(valid_path_1)

    valid_path1 = [os.path.join(valid_path_1, name) for name in valid_path1]

    valid_path_2 = '/scratch/eliransc/rnn_data/testset2_batches/0_0_0.6'  # '/scratch/eliransc/rnn_data/gt_gt_1_batches_G4_experiment'
    valid_path2 = os.listdir(valid_path_2)

    valid_path2 = [os.path.join(valid_path_2, name) for name in valid_path2]

    valid_path_3 = '/scratch/eliransc/rnn_data/testset2_batches/2_2_0.6'  # '/scratch/eliransc/rnn_data/gt_gt_1_batches_G4_experiment'
    valid_path3 = os.listdir(valid_path_3)

    valid_path3 = [os.path.join(valid_path_3, name) for name in valid_path3]

    valid_path_4 = '/scratch/eliransc/rnn_data/testset2_batches/1_1_0.6'  # '/scratch/eliransc/rnn_data/gt_gt_1_batches_G4_experiment'
    valid_path4 = os.listdir(valid_path_4)

    valid_path4 = [os.path.join(valid_path_4, name) for name in valid_path4]

    valid_path_5 = '/scratch/eliransc/rnn_data/testset2_batches/3_3_0.6'  # '/scratch/eliransc/rnn_data/gt_gt_1_batches_G4_experiment'
    valid_path5 = os.listdir(valid_path_5)

    valid_path5 = [os.path.join(valid_path_5, name) for name in valid_path5]

    # create dataset
    dataset = my_Dataset(data_paths)
    dataset_valid = my_Dataset(valid_path)
    dataset_valid1 = my_Dataset(valid_path1)
    dataset_valid2 = my_Dataset(valid_path2)
    dataset_valid3 = my_Dataset(valid_path3)
    dataset_valid4 = my_Dataset(valid_path4)
    dataset_valid5 = my_Dataset(valid_path5)

    # Hyper-parameters
    # input_size = 784 # 28x28

    num_epochs = 40
    lr_change = 1.025  # np.random.choice([1.025, 1.05, 1.1, 1.25, 1.3],  p=[0.2, 0.2, 0.2, 0.2, 0.2])
    batch_size = int(np.random.choice([1, 2, 4, 8], p=[0.25, 0.35, 0.25, 0.15]))
    learning_rate = np.random.choice([0.001, 0.005, 0.0005, 0.0002, 0.0001], p=[0.2, 0.2, 0.2, 0.2, 0.2])

    input_size = 30 + 2 * num_moments
    hidden_size = 128  # int(np.random.choice([32,64,128],  p=[0.3, 0.4, 0.3]))
    num_layers = np.random.randint(2, 6)

    loss_option = np.random.randint(1 , 3) # if 1 loss_queue if 2 loss_queue1

    sequence_length = args.time_ub

    output_size = 51

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

    valid_loader = DataLoader(dataset=dataset_valid,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0)

    valid_loader1 = DataLoader(dataset=dataset_valid1,
                               batch_size=1,
                               shuffle=True,
                               num_workers=0)

    valid_loader2 = DataLoader(dataset=dataset_valid2,
                               batch_size=1,
                               shuffle=True,
                               num_workers=0)

    valid_loader3 = DataLoader(dataset=dataset_valid3,
                               batch_size=1,
                               shuffle=True,
                               num_workers=0)

    valid_loader4 = DataLoader(dataset=dataset_valid4,
                               batch_size=1,
                               shuffle=True,
                               num_workers=0)

    valid_loader5 = DataLoader(dataset=dataset_valid5,
                               batch_size=1,
                               shuffle=True,
                               num_workers=0)



    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)

    model1 = RNN1(input_size, hidden_size, num_layers, output_size)  # .to(device)

    m = nn.Softmax(dim=2)

    num_epochs = 40
    n_total_steps = int(len(train_loader) / 50)
    loss_list = []
    setting_string = 'batch_size_' + str(batch_size * 16) + '_num_layers_' + str(num_layers) + '_num_epochs_' + str(
        num_epochs) + '_learning_rate_' + str(learning_rate) + '_hidden_size_' + str(hidden_size) + '_lr_change_' + str(
        lr_change+ '_loss_option_'+ str(loss_option))
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
                if loss_option == 1:
                    loss = loss_queue(soft, labels)
                else:
                    loss = loss_queue1(soft, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())


            else:
                print('Bad batch: ', i)

        print('Computing valid batch...')

        learning_rate = learning_rate ** lr_change

        torch.save(model.state_dict(),
                   '/scratch/eliransc/RNN_models_new/pytorch_gt_gt_1_true_moms_new_data_' + setting_string + '_' + str(
                       current_time) + '.pkl')

        model1.load_state_dict(
            torch.load('/scratch/eliransc/RNN_models_new/pytorch_gt_gt_1_true_moms_new_data_' + setting_string + '_' + str(
                current_time) + '.pkl'))  # ,map_location=torch.device('cude')


        valids = []
        for loader in [valid_loader, valid_loader1, valid_loader2, valid_loader3, valid_loader4, valid_loader5]:

            totloss = []

            for i, (inputs, labels) in tqdm(enumerate(loader)):

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

            valids.append(torch.tensor(totloss).mean())
            print(torch.tensor(totloss).mean())

        valid_loss_list.append(valids)

        pkl.dump((loss_list, valid_loss_list),
                 open('/scratch/eliransc/RNN_loss_vals_new/' + 'loss_' + setting_string + '_' + str(current_time) + '.pkl',
                      'wb'))


def parse_arguments(argv):


    parser = argparse.ArgumentParser()

    parser.add_argument('--time_ub', type=int, help='num sequences in a single sim', default=60)
    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)