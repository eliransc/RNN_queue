import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle as pkl
import os
from fastai.vision.all import *
import numpy as np
import os
from tqdm import tqdm
import argparse
#


def prepere_data(args):
    files = os.listdir(args.load_path)

    probs_ub = args.probs_ub
    input_result_all = np.array([])
    output_result_all = np.array([])

    for file in tqdm(files[args.LB: args.UB]):
        res_input = np.array([])
        mean_queue_arr = np.array([])
        time_dict, arrival_rates, model_inputs = pkl.load(open(os.path.join(args.load_path, file), 'rb'))
        for t in range(len(time_dict)):
            arr_input = np.concatenate((model_inputs[2], np.array([t]), np.array([arrival_rates[t]])), axis=0)
            arr_input = arr_input.reshape(1, arr_input.shape[0])
            probs = time_dict[t] / time_dict[t].sum()
            first_probs = probs[:probs_ub]
            complentry = 1 - np.sum(first_probs)
            probs_outout = np.append(first_probs, np.array([complentry]), axis=0).reshape(1, probs_ub + 1)
            if res_input.shape[0] == 0:
                res_input = arr_input
                probs_output_arr = probs_outout
            else:
                res_input = np.concatenate((res_input, arr_input), axis=0)
                probs_output_arr = np.concatenate((probs_output_arr, probs_outout), axis=0)

        single_input = res_input.reshape(1, res_input.shape[0], res_input.shape[1])
        single_output = probs_output_arr.reshape(1, probs_output_arr.shape[0], probs_output_arr.shape[1])
        if input_result_all.shape[0] == 0:
            input_result_all = single_input
            output_result_all = single_output
        else:
            input_result_all = np.concatenate((input_result_all, single_input), axis=0)
            output_result_all = np.concatenate((output_result_all, single_output), axis=0)

    pkl.dump(pkl.dump((input_result_all, output_result_all),
                      open(os.path.join(args.dump_path,'RNN_trainsient_traininig_prob_' + str(args.LB) + '_' + str(args.UB) + '.pkl'), 'wb')))


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--probs_ub', type=int, help='path to pkl folder', default=99)
    parser.add_argument('--LB', type=int, help='path to pkl folder', default=0)
    parser.add_argument('--UB', type=int, help='path to pkl folder', default=2000)
    parser.add_argument('--dump_path', type=str, help='path to pkl folder', default='/scratch/eliransc/pkl_data')
    parser.add_argument('--load_path', type=str, help='path to pkl folder', default='/scratch/eliransc/time_dependant')
    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    prepere_data(args)