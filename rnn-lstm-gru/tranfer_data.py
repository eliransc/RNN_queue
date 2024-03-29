import numpy as np
import pandas as pd
import pickle as pkl
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from fastai.vision.all import *
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle as pkl
from fastai.vision.all import *
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm
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


def main():

    list_path = 'list_batch_numbers15.pkl'

    if not os.path.exists(list_path):
        pkl.dump([], open(list_path, 'wb'))

    folder_path = '/scratch/eliransc/rnn_data_round2'

    tot_list = list(np.arange(1, len(os.listdir(folder_path))+1))

    num_list = pkl.load(open(list_path, 'rb'))

    temp3 = [x for x in tot_list if x not in num_list]
    ind = np.random.randint(len(temp3))

    folder_num = temp3[ind]

    num_list.append(folder_num)
    pkl.dump(num_list, open(list_path, 'wb'))

    batch_size = 32

    path = os.path.join(folder_path, 'folder_' + str(folder_num))

    files = os.listdir(path)
    num_batches = int(len(files) / batch_size)

    for curr_batch in tqdm(range(num_batches)):
        batch_input = np.array([])
        batch_output = np.array([])
        for ind in range(curr_batch * batch_size, (curr_batch + 1) * batch_size):

            try:
                res_input, prob_queue_arr = pkl.load(open(os.path.join(path, files[ind]), 'rb'))
                res_input = res_input.reshape(1, res_input.shape[0], res_input.shape[1])
                prob_queue_arr = prob_queue_arr.reshape(1, prob_queue_arr.shape[0], prob_queue_arr.shape[1])
                if ind == curr_batch * batch_size:
                    batch_input = res_input
                    batch_output = prob_queue_arr
                else:
                    batch_input = np.concatenate((batch_input, res_input), axis=0)
                    batch_output = np.concatenate((batch_output, prob_queue_arr), axis=0)
                # print(os.path.join(path, files[ind]))
                # os.remove(os.path.join(path, files[ind]))
            except:
                print('could not open file')
        batch_num = np.random.randint(1000, 100000000)
        if batch_input.shape[0] == 32:
            pkl.dump((batch_input, batch_output),
                     open(os.path.join('/scratch/eliransc/new_gt_g_1_batches2', 'batch2_' + str(batch_num) + '.pkl'),
                          'wb'))



if __name__ == '__main__':
    main()