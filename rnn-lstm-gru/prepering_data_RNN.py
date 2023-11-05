# imports
import numpy as np
import pandas as pd
import pickle as pkl
import os
import matplotlib.pyplot as plt
import argparse
import torch
import pickle as pkl
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm
import sys


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

    if not 'C:' in os.getcwd().split('/')[0]:

        curr_path = '/home/eliransc/projects/def-dkrass/eliransc/RNN_queue/rnn-lstm-gru'

        files_in_path = os.listdir(curr_path)

        server_file = [file for file in files_in_path if '_server' in file]

        server_name = server_file[0].split('_')[0]


    if 'C:' in os.getcwd().split('/')[0]:
        path = r'C:\Users\user\workspace\data\mt_g_1'
    else:
        path = '/scratch/eliransc/gt_g_1_data'

    # path = '/scratch/eliransc/gt_g_1_data'
    # files = os.listdir(path)
    # for file in tqdm(files):
    #     try:
    #         res_input, prob_queue_arr = pkl.load(open(os.path.join(path, file), 'rb'))
    #     except:
    #         print(file)
    #         os.remove(os.path.join(path, file))

    path = '/scratch/eliransc/gt_g_1_data2'
    files = os.listdir(path)
    # files_rho_groups = os.listdir()

    # for rho in files_rho_groups:
    #     curr_path = os.path.join(path, rho)
    #     files = os.listdir(curr_path)

    if 'C:' in os.getcwd().split('/')[0]:

        if os.path.exists('./pkl/df_files.pkl'):
            df_files = pkl.load(open('./pkl/df_files.pkl'))
        else:
            df_files = pd.DataFrame([], columns=['file', 'batch'])

    else:

        if os.path.exists('./pkl/df_files.pkl'):
            df_files = pkl.load(open('./pkl/df_files.pkl'))
        else:
            df_files = pd.DataFrame([], columns=['file', 'batch'])

    for file in files:
        try:
            res_input, prob_queue_arr = pkl.load(open(os.path.join(path, file), 'rb'))
        except:
            os.remove(os.path.join(path, file))
    files = os.listdir(path)
    batch_size = 16
    num_batches = int(len(files) / batch_size)

    for curr_batch in tqdm(range(num_batches)):
        batch_input = np.array([])
        batch_output = np.array([])

        for ind in range(curr_batch * batch_size, (curr_batch + 1) * batch_size):
            curr_file = files[ind]
            curr_ind_df = df_files.shape[0]
            df_files.loc[curr_ind_df, 'file'] = curr_file
            df_files.loc[curr_ind_df, 'batch'] = curr_batch

            pkl.dump(df_files, open('df_files.pkl', 'wb'))
            res_input, prob_queue_arr = pkl.load(open(os.path.join(path, files[ind]), 'rb'))

            # res_input, prob_queue_arr = create_single_data_point(path, files, ind)
            res_input = res_input.reshape(1, res_input.shape[0], res_input.shape[1])
            prob_queue_arr = prob_queue_arr.reshape(1, prob_queue_arr.shape[0], prob_queue_arr.shape[1])
            if ind == curr_batch * batch_size:
                batch_input = res_input
                batch_output = prob_queue_arr
            else:
                batch_input = np.concatenate((batch_input, res_input), axis=0)
                batch_output = np.concatenate((batch_output, prob_queue_arr), axis=0)

        path_dump = '/scratch/eliransc/rnn_data/gt_gt_1_batches_G4_experiment/'

        # curr_dump_path = os.path.join(path_dump, rho)

        # if not curr_dump_path:
        #     os.mkdir(curr_dump_path)

        pkl.dump((batch_input, batch_output), open(
             os.path.join(path_dump, server_name +'_'+ str(curr_batch) + 'const_arrival_dist.pkl'), 'wb'))


def parse_arguments(argv):


    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=bool, help='the path of the average waiting time', default= '/scratch/eliransc/time_dependant_cyclic')
    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)