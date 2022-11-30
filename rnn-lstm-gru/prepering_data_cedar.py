# imports
import numpy as np
import pandas as pd
import pickle as pkl
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from fastai.vision.all import *
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle as pkl
from fastai.vision.all import *
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm



def create_single_data_point(path, files_list, ind):
    res_input = np.array([])
    prob_queue_arr = np.array([])
    time_dict, arrival_rates, model_inputs, initial = pkl.load(open(os.path.join(path, files_list[ind]), 'rb'))
    for t in range(len(time_dict)):
        arr_input = np.concatenate((np.log(model_inputs[2][:5]), np.array([t]), np.array([arrival_rates[t]]), initial[:5]), axis =0)
        arr_input = arr_input.reshape(1,arr_input.shape[0])
        probs = (time_dict[t]/time_dict[t].sum())
        fifty_or_more = probs[50:].sum()
        probs_output = np.concatenate((probs[:50], np.array([fifty_or_more]) ), axis = 0)
        if res_input.shape[0] == 0:
            res_input = arr_input
            prob_queue_arr = np.array(probs_output).reshape(1,probs_output.shape[0])
        else:
            res_input = np.concatenate((res_input, arr_input), axis = 0)
            prob_queue_arr = np.concatenate((prob_queue_arr, np.array(probs_output).reshape(1,probs_output.shape[0])), axis = 0)
    return (res_input, prob_queue_arr)



def main(args):

    path = '/scratch/eliransc/time_dependant_cyclic'
    files = os.listdir(path)

    df_files = pd.DataFrame([], columns=['file', 'batch'])

    batch_size = 64
    num_batches = int(len(files) / batch_size)

    for curr_batch in tqdm(range(num_batches)):
        batch_input = np.array([])
        batch_output = np.array([])

        for ind in range(curr_batch * batch_size, (curr_batch + 1) * batch_size):
            curr_file = files[ind]
            curr_ind_df = df_files.shape[0]
            df_files.loc[curr_ind_df, 'file'] = curr_file
            df_files.loc[curr_ind_df, 'batch'] = curr_batch

            pkl.dump(df_files, open('cedar_df_files.pkl', 'wb'))
            res_input, prob_queue_arr = create_single_data_point(path, files, ind)
            res_input = res_input.reshape(1, res_input.shape[0], res_input.shape[1])
            prob_queue_arr = prob_queue_arr.reshape(1, prob_queue_arr.shape[0], prob_queue_arr.shape[1])
            if ind == curr_batch * batch_size:
                batch_input = res_input
                batch_output = prob_queue_arr
            else:
                batch_input = np.concatenate((batch_input, res_input), axis=0)
                batch_output = np.concatenate((batch_output, prob_queue_arr), axis=0)

        pkl.dump((batch_input, batch_output), open(
            '/scratch/eliransc/pkl_rnn/Cedar_batch_' + str(curr_batch) + '.pkl', 'wb'))


def parse_arguments(argv):


    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=bool, help='the path of the average waiting time', default= '/scratch/eliransc/time_dependant_cyclic')
    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)