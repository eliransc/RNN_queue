# imports
import simpy
import numpy as np
import sys
import argparse
import pandas as pd
import pickle as pkl
import time
import os
from tqdm import tqdm
from datetime import datetime
import random
import pickle as pkl
import matplotlib.pyplot as plt
import sympy
from sympy import *
from scipy.special import gamma, factorial
import random
import time
import itertools


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\user\workspace\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')

import os
import random
import pandas as pd
import argparse
from tqdm import tqdm
from butools.ph import *
from butools.map import *
from butools.queues import *
import time
from butools.mam import *
from butools.dph import *
from scipy.linalg import expm, sinm, cosm

from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
# import seaborn as sns
import random
from scipy.stats import loguniform
# from butools.fitting import *
from datetime import datetime
# from fastbook import *
import itertools
from scipy.special import factorial

import pickle as pkl





def ser_mean(alph, T):
    e = np.ones((T.shape[0], 1))
    try:
        return -np.dot(np.dot(alph, np.linalg.inv(T)), e)
    except:
        return False


def compute_pdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return np.dot(np.dot(s, expm(A * x)), A0)


def compute_cdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return 1 - np.sum(np.dot(s, expm(A * x)))


def gives_rate(states_inds, rate, ph_size):
    '''
    states_ind: the out states indices
    rate: the total rate out
    return: the out rate array from that specific state
    '''
    final_rates = np.zeros(ph_size - 1)  ## initialize the array
    rands_weights_out_rate = np.random.rand(states_inds.shape[0])  ## Creating the weights of the out rate
    ## Computing the out rates
    final_rates[states_inds] = (rands_weights_out_rate / np.sum(rands_weights_out_rate)) * rate
    return final_rates


def create_row_rates(row_ind, is_absorbing, in_rate, non_abrosing_out_rates, ph_size, non_absorbing):
    '''
    row_ind: the current row
    is_abosboing: true if it an absorbing state
    in_rate: the rate on the diagonal
    non_abrosing_out_rates: the matrix with non_abrosing_out_rates
    ph_size: the size of phase type
    return: the ph row_ind^th of the ph matrix
    '''

    finarr = np.zeros(ph_size)
    finarr[row_ind] = -in_rate  ## insert the rate on the diagonal with a minus sign
    if is_absorbing:  ## no further changes is requires
        return finarr
    else:
        all_indices = np.arange(ph_size)
        all_indices = all_indices[all_indices != row_ind]  ## getting the non-diagonal indices
        rate_ind = np.where(non_absorbing == row_ind)  ## finding the current row in non_abrosing_out_rates
        finarr[all_indices] = non_abrosing_out_rates[rate_ind[0][0]]
        return finarr


def give_s_A_given_size(ph_size):
    '''
    generate general PH given size
    '''

    potential_vals = np.linspace(0.1, 10, 20000)
    randinds = np.random.randint(potential_vals.shape[0], size=ph_size)
    ser_rates = (potential_vals[randinds]).reshape((1, ph_size))
    w = np.random.rand(ph_size + 1)
    numbers = np.arange(ph_size + 1)  # an array from 0 to ph_size + 1
    distribution = w / np.sum(w)  ## creating a pdf from the weights of w
    random_variable = rv_discrete(values=(numbers, distribution))  ## constructing a python pdf
    ww = random_variable.rvs(size=1)
    ## choosing the states that are absorbing
    absorbing_states = np.sort(np.random.choice(ph_size, ww[0], replace=False))
    non_absorbing = np.setdiff1d(np.arange(ph_size), absorbing_states, assume_unique=True)

    N = ph_size - ww[0]  ## N is the number of non-absorbing states
    p = np.random.rand()  # the probability that a non absorbing state is fully transient
    mask_full_trans = np.random.choice([True, False], size=N, p=[p, 1 - p])  # True if row sum to 0
    ser_rates = ser_rates.flatten()

    ## Computing the total out of state rate, if absorbing, remain the same
    p_outs = np.random.rand(N)  ### this is proportional rate out
    orig_rates = ser_rates[non_absorbing]  ## saving the original rates
    new_rates = orig_rates * p_outs  ## Computing the total out rates
    out_rates = np.where(mask_full_trans, orig_rates, new_rates)  ## Only the full trans remain as the original

    ## Choosing the number of states that will have a postive rate out for every non-absorbing state
    num_trans_states = np.random.randint(1, ph_size, N)

    ## Choosing which states will go from each non-absorbing state
    trans_states_list = [np.sort(np.random.choice(ph_size - 1, num_trans_states[j], replace=False)) for j in range(N)]
    # Computing out rates
    non_abrosing_out_rates = [gives_rate(trans_states, out_rates[j], ph_size) for j, trans_states in
                              enumerate(trans_states_list)]
    ## Finalizing the matrix

    #     return trans_states_list, absorbing_states, ser_rates, non_abrosing_out_rates
    lists_rate_mat = [
        create_row_rates(row_ind, row_ind in absorbing_states, ser_rates[row_ind], non_abrosing_out_rates, ph_size,
                         non_absorbing) for row_ind in range(ph_size)]
    A = np.concatenate(lists_rate_mat).reshape((ph_size, ph_size))  ## converting all into one numpy array

    num_of_pos_initial_states = np.random.randint(1, ph_size + 1)
    non_zero_probs = np.random.dirichlet(np.random.rand(num_of_pos_initial_states), 1)
    inds_of_not_zero_probs = np.sort(np.random.choice(ph_size, num_of_pos_initial_states, replace=False))
    s = np.zeros(ph_size)
    s[inds_of_not_zero_probs] = non_zero_probs

    return (s, A)



def create_erlang_row(rate, ind, size):
    'Compute a single Erlang row'
    aa = np.zeros(size)
    aa[ind] = -rate
    if ind < size - 1:
        aa[ind + 1] = rate
    return aa

def ser_moment_n(s, A, mom):
    '''
    Compute the  'mom' moment of PH
    '''
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) *factorial(mom)*np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False

def compute_first_n_moments(s, A, n=3):
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list


def compute_cdf_within_range(x_vals, s, A):
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_cdf(x, s, A).flatten())

    return pdf_list


def compute_pdf_within_range(x_vals, s, A):
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_pdf(x, s, A).flatten())

    return pdf_list


def recursion_group_size(group_left, curr_vector, phases_left):
    if group_left == 1:
        return np.append(phases_left, curr_vector)
    else:

        if phases_left + 1 - group_left == 1:
            curr_size = 1
        else:
            curr_size = np.random.randint(1, phases_left + 1 - group_left)
        return recursion_group_size(group_left - 1, np.append(curr_size, curr_vector), phases_left - curr_size)





def generate_erlang_given_rates(rate, ph_size):
    A = np.identity(ph_size)
    A_list = [create_erlang_row(rate, ind, ph_size) for ind in range(ph_size)]
    A = np.concatenate(A_list).reshape((ph_size, ph_size))
    return A


def find_when_cdf_cross_1(x, y):
    if y[-1] < 0.9999:
        return False
    for ind in range(len(y)):
        if y[ind] > 0.9999:
            return ind
    return False


def find_normalizing_const(s, A, x, itera=0, thrsh=0.9999):
    if itera > 50:
        return False
    curr_cdf = compute_cdf(x, s, A).flatten()[0]

    if curr_cdf < thrsh:
        return find_normalizing_const(s, A, x * 2, itera + 1, thrsh)
    elif (curr_cdf > thrsh) and (curr_cdf < 1.):
        return x
    else:
        return find_normalizing_const(s, A, x / 2, itera + 1, thrsh)


def normalize_matrix(s, A):
    normalize = find_normalizing_const(s, A, 6)
    if normalize > 1:
        A = A * normalize
    return (A, s)


def compute_R(lam, alph, T):
    e = torch.ones((T.shape[0], 1))
    return np.array(lam * torch.inverse(lam * torch.eye(T.shape[0]) - lam * e @ alph - T))


from numpy.linalg import matrix_power


def steady_i(rho, alph, R, i):
    return (1 - rho) * alph @ matrix_power(R, i)


def ser_mean(alph, T):
    e = torch.ones((T.shape[0], 1))
    try:
        return -alph @ torch.inverse(T) @ e
    except:
        return False


def create_final_x_data(s, A, lam):

    lam_arr = np.zeros((A.shape[0] + 1, 1))

    s1 = s.reshape((1, s.shape[0]))
    expect_ser = ser_moment_n(s, A, 1)
    if expect_ser:
        #         expect_ser = expect_ser[0][0]
        # mu = 1/expect_ser
        # lam = np.random.uniform(0.3*mu, 0.9*mu, 1)[0]
        # lam = lam * 0.95
        lam_arr[0, 0] = lam


        return np.append(np.append(A, s1, axis=0), lam_arr, axis=1).astype(np.float32)



def compute_y_data_given_folder(x, ph_size_max, tot_prob=500, eps=0.00001):
    try:
        lam = x[0, ph_size_max].item()
        A = x[:ph_size_max, :ph_size_max]
        s = x[ph_size_max, :ph_size_max].reshape((1, ph_size_max))
        expect_ser = ser_moment_n(s, A, 1)
        if expect_ser:
            rho = lam * expect_ser[0][0]

            R = compute_R(lam, s, A)

            steady_state = np.array([1 - rho])
            for i in range(1, tot_prob-1):
                steady_state = np.append(steady_state, np.sum(steady_i(rho, s, R, i)))

            steady_state = np.append(steady_state, 1-np.sum(steady_state))
            return steady_state


    except:
        print("x is not valid")


def create_short_tale(group_sizes, rates, probs):
    erlang_list = [generate_erlang_given_rates(rates[ind], ph_size) for ind, ph_size in enumerate(group_sizes)]
    ph_size = np.sum(group_sizes)
    final_a = np.zeros((ph_size, ph_size))
    final_s = np.zeros(ph_size)
    for ind in range(group_sizes.shape[0]):
        final_s[np.sum(group_sizes[:ind])] = probs[ind]
        final_a[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
        np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = erlang_list[ind]
    return final_s, final_a


def get_lower_upper_x(phsize, rate, prob_limit=0.999):
    lower_flag = False
    A = generate_erlang_given_rates(rate, phsize)
    s = np.zeros(phsize)
    s[0] = 1
    x_vals = np.linspace(0, 1, 300)
    for x in x_vals:
        if not lower_flag:
            if compute_cdf(x, s, A) > 0.001:
                lower = x
                lower_flag = True

        if compute_cdf(x, s, A) > prob_limit:
            upper = x

            return (lower, upper, phsize, rate)

    return False



def give_rates_given_Er_sizes(df_, sizes, ratio_size):
    rates = np.array([])
    ratio_list = list(np.arange(ratio_size))
    for ph_size in sizes:
        curr_ratio = random.choice(ratio_list)
        ratio_list.remove(curr_ratio)
        inds = df_.loc[df_['phases'] == ph_size, :].index

        rates = np.append(rates, df_.loc[inds[curr_ratio], 'rate'])

    return rates


def create_rate_phsize_combs(vals_bound):
    all_combs_list = []
    for size in vals_bound.keys():
        curr_list = [(size, vals_bound[size] * ratios_rates[ind_rate]) for ind_rate, rate in enumerate(ratios_rates)]
        all_combs_list.append(curr_list)
    return all_combs_list




def find_upper_bound_rate_given_n(n, upper_bound):
    if upper_bound == 0:
        return False
    if np.array(get_lower_upper_x(n, upper_bound)).any():
        return find_upper_bound_rate_given_n(n, upper_bound - 1)
    else:
        return upper_bound + 1




def find_when_cdf_cross_0_999(s, A, x, itera=0, thrsh=0.9995):
    curr_cdf = compute_cdf(x, s, A).flatten()[0]
    if itera > 50:
        if curr_cdf > 0.999:
            return x
        else:
            return False

    if curr_cdf < thrsh:
        return find_when_cdf_cross_0_999(s, A, x * 2, itera + 1, thrsh)
    elif (curr_cdf > thrsh) and (curr_cdf < 1.):
        return x
    else:
        return find_when_cdf_cross_0_999(s, A, x / 2, itera + 1, thrsh)


def normalize_ph_so_it_1_when_cdf_1(s, A, initial_val=0.5):
    norm_const = find_when_cdf_cross_0_999(s, A, initial_val)
    if norm_const == 0:
        print('Not able to find normalizing constant')
        return False
    else:
        A = A * norm_const

    return (s, A)



def saving_batch(x_y_data, data_path, data_sample_name, num_moms, save_x = False):
    '''

    :param x_y_data: the data is a batch of tuples: ph_input, first num_moms moments and steady-state probs
    :param data_path: the folder in which we save the data
    :param data_sample_name: the name of file
    :param num_moms: number of moments we compute
    :param save_x: should we save ph_data
    :return:
    '''

    now = datetime.now()


    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])
    x_list =  []
    mom_list = []
    y_list = []

    for x_y in x_y_data:
        if type(x_y) != bool:
            if save_x:
                x_list.append(torch.from_numpy(x_y[0]))
            mom_list.append(torch.from_numpy(x_y[0]))
            y_list.append(torch.from_numpy(x_y[1]))


    if save_x: # should we want to save the x_data
        # x_list = [torch.from_numpy(x_y[0]) for x_y in x_y_data if type(x_y) != bool]
        # torch_x = torch.stack(x_list).float()
        pkl_name_xdat = 'xdat_' + data_sample_name + current_time +'size_' + '.pkl' #+ str(torch_x.shape[0]) +
        full_path_xdat = os.path.join(data_path, pkl_name_xdat)
        pkl.dump(x_list, open(full_path_xdat, 'wb'))

    # dumping moments
    # mom_list = [torch.from_numpy(x_y[1]) for x_y in x_y_data if type(x_y) != bool]
    torch_moms = torch.stack(mom_list).float()
    pkl_name_moms = 'moms_' + str(num_moms) + data_sample_name + current_time + 'size_'+ str(torch_moms.shape[0]) + '.pkl'
    full_path_moms = os.path.join(data_path, pkl_name_moms)
    pkl.dump(torch_moms, open(full_path_moms, 'wb'))


    # dumping steady_state
    # y_list = [torch.from_numpy(x_y[2]) for x_y in x_y_data if type(x_y) != bool]
    torch_y = torch.stack(y_list).float()
    pkl_name_ydat = 'ydat_' + data_sample_name + current_time +'size_'+ str(torch_y.shape[0]) + '.pkl'
    full_path_ydat = os.path.join(data_path, pkl_name_ydat)
    pkl.dump(torch_y, open(full_path_ydat, 'wb'))


def give_s_A_given__fixed_size(ph_size, scale_low, scale_high):
    if ph_size > 1:
        potential_vals = np.linspace(scale_low, scale_high, 20000)
        randinds = np.random.randint(potential_vals.shape[0], size=ph_size)
        ser_rates = (potential_vals[randinds]).reshape((1, ph_size))
        w = np.random.rand(ph_size)
        numbers = np.arange(0, ph_size + 1)  # an array from 0 to ph_size + 1
        p0 = 0.9
        distribution = (w / np.sum(w)) * (1 - p0)  ## creating a pdf from the weights of w
        distribution = np.append(p0, distribution)
        random_variable = rv_discrete(values=(numbers, distribution))  ## constructing a python pdf
        ww = random_variable.rvs(size=1)

        ## choosing the states that are absorbing
        absorbing_states = np.sort(np.random.choice(ph_size, ww[0], replace=False))
        non_absorbing = np.setdiff1d(np.arange(ph_size), absorbing_states, assume_unique=True)

        N = ph_size - ww[0]  ## N is the number of non-absorbing states
        p = np.random.rand()  # the probability that a non absorbing state is fully transient
        mask_full_trans = np.random.choice([True, False], size=N, p=[p, 1 - p])  # True if row sum to 0
        if np.sum(mask_full_trans) == mask_full_trans.shape[0]:
            mask_full_trans = False
        ser_rates = ser_rates.flatten()

        ## Computing the total out of state rate, if absorbing, remain the same
        p_outs = np.random.rand(N)  ### this is proportional rate out
        orig_rates = ser_rates[non_absorbing]  ## saving the original rates
        new_rates = orig_rates * p_outs  ## Computing the total out rates
        out_rates = np.where(mask_full_trans, orig_rates, new_rates)  ## Only the full trans remain as the original

        ## Choosing the number of states that will have a postive rate out for every non-absorbing state

        num_trans_states = np.random.randint(1, ph_size, N)

        ## Choosing which states will go from each non-absorbing state
        trans_states_list = [np.sort(np.random.choice(ph_size - 1, num_trans_states[j], replace=False)) for j in
                             range(N)]
        # Computing out rates
        non_abrosing_out_rates = [gives_rate(trans_states, out_rates[j], ph_size) for j, trans_states in
                                  enumerate(trans_states_list)]
        ## Finalizing the matrix

        #     return trans_states_list, absorbing_states, ser_rates, non_abrosing_out_rates
        lists_rate_mat = [
            create_row_rates(row_ind, row_ind in absorbing_states, ser_rates[row_ind], non_abrosing_out_rates, ph_size,
                             non_absorbing) for row_ind in range(ph_size)]
        A = np.concatenate(lists_rate_mat).reshape((ph_size, ph_size))  ## converting all into one numpy array

        num_of_pos_initial_states = np.random.randint(1, ph_size + 1)
        non_zero_probs = np.random.dirichlet(np.random.rand(num_of_pos_initial_states), 1)
        inds_of_not_zero_probs = np.sort(np.random.choice(ph_size, num_of_pos_initial_states, replace=False))
        s = np.zeros(ph_size)
        s[inds_of_not_zero_probs] = non_zero_probs

    else:
        s = np.array([1.])
        potential_vals = np.linspace(scale_low, scale_high, 20000)
        randinds = np.random.randint(potential_vals.shape[0], size=ph_size)
        ser_rates = (potential_vals[randinds]).reshape((1, ph_size))
        A = -ser_rates

    return (s, A)


def create_mix_erlang_ph(scale_low=1, max_scale_high=15, max_ph=500):

    erlang_max_size = np.random.randint(int(0.25 * max_ph), int(0.75 * max_ph))
    scale_high = np.random.uniform(2, max_scale_high)
    ph_size_gen_ph = np.random.randint(5, max_ph - erlang_max_size)
    num_groups = np.random.randint(1, min(20, ph_size_gen_ph - 1))
    group_sizes = np.random.randint(1, 25, num_groups)

    group_sizes_gen_ph = (group_sizes * ph_size_gen_ph / np.sum(group_sizes)).astype(int) + 1
    erlang_list_gen_ph = [give_s_A_given__fixed_size(size, scale_low, scale_high) for size in group_sizes_gen_ph]
    erlang_list_gen_ph_A = [lis[1] for lis in erlang_list_gen_ph]
    erlang_list_gen_ph_s = [lis[0] for lis in erlang_list_gen_ph]

    ph_size_erl = np.random.randint(5, erlang_max_size)
    num_groups = np.random.randint(2, min(30, ph_size_erl - 1))
    group_sizes = np.random.randint(1, 25, num_groups)

    rates = ((np.ones(num_groups) * np.random.uniform(1, 1.75)) ** np.arange(num_groups))
    group_sizes_erl = (group_sizes * ph_size_erl / np.sum(group_sizes)).astype(int) + 1
    erlang_list_erl = [generate_erlang_given_rates(rates[ind], ph_size_erl) for ind, ph_size_erl in
                       enumerate(group_sizes_erl)]
    group_sizes = np.append(group_sizes_gen_ph, group_sizes_erl)

    rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)

    ph_list = erlang_list_gen_ph_A + erlang_list_erl

    ph_size = np.sum(group_sizes)
    A = np.zeros((ph_size, ph_size))
    s = np.zeros(ph_size)
    for ind in range(group_sizes.shape[0]):
        if ind < group_sizes_gen_ph.shape[0]:
            s[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = rand_probs[0][ind] * \
                                                                                        erlang_list_gen_ph_s[ind]
        else:
            s[np.sum(group_sizes[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
        A[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
        np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = ph_list[ind]

    fst_mom = compute_first_n_moments(s, A, 1)
    if type(fst_mom[0]) != bool:
        A = A * fst_mom[0][0]
        fst_mom = compute_first_n_moments(s, A, 1)
        if (fst_mom[0] > 0.99999) & (fst_mom[0] < 1.000001):
            #         A = A*give_cdf_1_1_norm_const(s, A)
            return (s, A)
        else:
            return False
    else:
        return False


def create_gen_erlang_many_ph(max_ph_size = 500):
    ph_size = np.random.randint(1, max_ph_size)
    num_groups = np.random.randint(2,20)
    group_sizes = np.random.randint(1,25,num_groups)
    group_sizes_1 = (group_sizes*ph_size/np.sum(group_sizes)).astype(int)+1
    rates = ((np.ones(num_groups)*np.random.uniform(1, 1.75))**np.arange(num_groups))
    s,A = create_gen_erlang_given_sizes(group_sizes_1, rates)

    A = A*compute_first_n_moments(s, A, 1)[0][0]
    return (s,A)

def create_gen_erlang_given_sizes(group_sizes, rates, probs=False):
    ph_size = np.sum(group_sizes)
    erlang_list = [generate_erlang_given_rates(rates[ind], ph_size) for ind, ph_size in enumerate(group_sizes)]
    final_a = np.zeros((ph_size, ph_size))
    final_s = np.zeros(ph_size)
    if type(probs) == bool:
        rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)
        rands = np.random.rand(group_sizes.shape[0])
        rand_probs = rands / np.sum(rands).reshape((1, rand_probs.shape[0]))
    else:
        rand_probs = probs
    for ind in range(group_sizes.shape[0]):
        final_s[np.sum(group_sizes[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
        final_a[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
        np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = erlang_list[ind]

    return final_s, final_a


def send_to_the_right_generator(num_ind, max_ph_size,  num_moms, data_path, data_sample_name):

    if num_ind == 1:
        s_A = create_mix_erlang_ph() # Classes of erlangs and non-erlangs
    elif num_ind > 1:
        s_A = create_gen_erlang_many_ph() # Classes of erlangs
    else:
        s_A = create_Erlang_given_ph_size(np.random.randint(1,max_ph_size))  # One large Erlang
    if type(s_A) != bool:
        try:
            s = s_A[0]
            A = s_A[1]

            return (s,A)


        except:
            print('Not able to extract s and A')

def compute_y_moms(s,A,num_moms,max_ph_size):

    lam_vals = np.random.uniform(0.8, 0.99, 1)

    lam_y_list = []

    for lam in lam_vals:
        x = create_final_x_data(s, A, lam)

        y = compute_y_data_given_folder(x, x.shape[0] - 1, tot_prob=500, eps=0.0001)
        if type(y) == np.ndarray:
            moms = compute_first_n_moments(s, A, num_moms)

            mom_arr = np.concatenate(moms, axis=0)

            lam = x[0, x.shape[0] - 1]
            mom_arr = np.log(mom_arr)
            mom_arr = np.delete(mom_arr, 0)
            mom_arr = np.append(lam, mom_arr)

            if not np.any(np.isinf(mom_arr)):

                lam_y_list.append((mom_arr, y))

    return lam_y_list


def generate_one_ph(batch_size, max_ph_size, num_moms, data_path, data_sample_name):

    elements = [1, 2, 3]
    probabilities = [0.495, 0.495, 0.01]
    sample_type_arr = np.random.choice(elements, batch_size, p=probabilities)
    x_y_moms_list = [send_to_the_right_generator(val, max_ph_size,  num_moms, data_path, data_sample_name) for val in sample_type_arr]
    x_y_moms_list = [x_y_moms for x_y_moms in x_y_moms_list if x_y_moms]
    x_y_moms_lists = [compute_y_moms(x_y_moms[0], x_y_moms[1], num_moms, max_ph_size) for x_y_moms  in x_y_moms_list]
    saving_batch(list(itertools.chain(*x_y_moms_lists)), data_path, data_sample_name, num_moms)

    return 1

def create_Erlang_given_ph_size(ph_size):
    '''
    Create Phase Type representation given size: ph_size
    '''
    s = np.zeros(ph_size)
    s[0] = 1
    rate = ph_size
    A = generate_erlang_given_rates(rate, ph_size)
    return (s,A)



def generate_erlangs(batch_size, max_ph_size, num_moms, data_path, data_sample_name):
    '''
    Sample a batch of Erlangs
    '''

    sizes = np.random.randint(500,max_ph_size,batch_size)
    x_y_moms_list = [create_Erlang_given_ph_size(ph_size) for ph_size in sizes]
    x_y_moms_list = [x_y_moms for x_y_moms in x_y_moms_list if x_y_moms]
    x_y_moms_lists = [compute_y_moms(x_y_moms[0],x_y_moms[1], num_moms, max_ph_size) for x_y_moms  in x_y_moms_list]
    saving_batch(list(itertools.chain(*x_y_moms_lists)), data_path, data_sample_name, num_moms)



    return 1


def create_gen_erlang_given_sizes(group_sizes, rates, probs=False):
    ph_size = np.sum(group_sizes)
    erlang_list = [generate_erlang_given_rates(rates[ind], ph_size) for ind, ph_size in enumerate(group_sizes)]
    final_a = np.zeros((ph_size, ph_size))
    final_s = np.zeros(ph_size)
    if type(probs) == bool:
        rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)
        rands = np.random.rand(group_sizes.shape[0])
        rand_probs = rands / np.sum(rands).reshape((1, rand_probs.shape[0]))
    else:
        rand_probs = probs
    for ind in range(group_sizes.shape[0]):
        final_s[np.sum(group_sizes[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
        final_a[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
        np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = erlang_list[ind]

    return final_s, final_a


def create_gen_erlang_many_ph(max_ph_size = 500):
    ph_size = np.random.randint(1, max_ph_size)
    num_groups = np.random.randint(2,20)
    group_sizes = np.random.randint(1,25,num_groups)
    group_sizes_1 = (group_sizes*ph_size/np.sum(group_sizes)).astype(int)+1
    rates = ((np.ones(num_groups)*np.random.uniform(1, 1.75))**np.arange(num_groups))
    s,A = create_gen_erlang_given_sizes(group_sizes_1, rates)

    A = A*compute_first_n_moments(s, A, 1)[0][0]
    return (s,A)


def ser_moment_n(s, A, mom):
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) *factorial(mom)*np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False

def compute_first_n_moments(s, A, n=3):
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list


def create_erlang_row(rate, ind, size):
    aa = np.zeros(size)
    aa[ind] = -rate
    if ind < size - 1:
        aa[ind + 1] = rate
    return aa


def create_row_rates(row_ind, is_absorbing, in_rate, non_abrosing_out_rates, ph_size, non_absorbing):
    '''
    row_ind: the current row
    is_abosboing: true if it an absorbing state
    in_rate: the rate on the diagonal
    non_abrosing_out_rates: the matrix with non_abrosing_out_rates
    ph_size: the size of phase type
    return: the ph row_ind^th of the ph matrix
    '''

    finarr = np.zeros(ph_size)
    finarr[row_ind] = -in_rate  ## insert the rate on the diagonal with a minus sign
    if is_absorbing:  ## no further changes is requires
        return finarr
    else:
        all_indices = np.arange(ph_size)
        all_indices = all_indices[all_indices != row_ind]  ## getting the non-diagonal indices
        rate_ind = np.where(non_absorbing == row_ind)  ## finding the current row in non_abrosing_out_rates
        finarr[all_indices] = non_abrosing_out_rates[rate_ind[0][0]]
        return finarr

def generate_erlang_given_rates(rate, ph_size):
    A = np.identity(ph_size)
    A_list = [create_erlang_row(rate, ind, ph_size) for ind in range(ph_size)]
    A = np.concatenate(A_list).reshape((ph_size, ph_size))
    return A

def gives_rate(states_inds, rate, ph_size):
    '''
    states_ind: the out states indices
    rate: the total rate out
    return: the out rate array from that specific state
    '''
    final_rates = np.zeros(ph_size - 1)  ## initialize the array
    rands_weights_out_rate = np.random.rand(states_inds.shape[0])  ## Creating the weights of the out rate
    ## Computing the out rates
    final_rates[states_inds] = (rands_weights_out_rate / np.sum(rands_weights_out_rate)) * rate
    return final_rates

def give_s_A_given__fixed_size(ph_size, scale_low, scale_high):
    if ph_size > 1:
        potential_vals = np.linspace(scale_low, scale_high, 20000)
        randinds = np.random.randint(potential_vals.shape[0], size=ph_size)
        ser_rates = (potential_vals[randinds]).reshape((1, ph_size))
        w = np.random.rand(ph_size)
        numbers = np.arange(0, ph_size + 1)  # an array from 0 to ph_size + 1
        p0 = 0.9
        distribution = (w / np.sum(w)) * (1 - p0)  ## creating a pdf from the weights of w
        distribution = np.append(p0, distribution)
        random_variable = rv_discrete(values=(numbers, distribution))  ## constructing a python pdf
        ww = random_variable.rvs(size=1)

        ## choosing the states that are absorbing
        absorbing_states = np.sort(np.random.choice(ph_size, ww[0], replace=False))
        non_absorbing = np.setdiff1d(np.arange(ph_size), absorbing_states, assume_unique=True)

        N = ph_size - ww[0]  ## N is the number of non-absorbing states
        p = np.random.rand()  # the probability that a non absorbing state is fully transient
        mask_full_trans = np.random.choice([True, False], size=N, p=[p, 1 - p])  # True if row sum to 0
        if np.sum(mask_full_trans) == mask_full_trans.shape[0]:
            mask_full_trans = False
        ser_rates = ser_rates.flatten()

        ## Computing the total out of state rate, if absorbing, remain the same
        p_outs = np.random.rand(N)  ### this is proportional rate out
        orig_rates = ser_rates[non_absorbing]  ## saving the original rates
        new_rates = orig_rates * p_outs  ## Computing the total out rates
        out_rates = np.where(mask_full_trans, orig_rates, new_rates)  ## Only the full trans remain as the original

        ## Choosing the number of states that will have a postive rate out for every non-absorbing state

        num_trans_states = np.random.randint(1, ph_size, N)

        ## Choosing which states will go from each non-absorbing state
        trans_states_list = [np.sort(np.random.choice(ph_size - 1, num_trans_states[j], replace=False)) for j in
                             range(N)]
        # Computing out rates
        non_abrosing_out_rates = [gives_rate(trans_states, out_rates[j], ph_size) for j, trans_states in
                                  enumerate(trans_states_list)]
        ## Finalizing the matrix

        #     return trans_states_list, absorbing_states, ser_rates, non_abrosing_out_rates
        lists_rate_mat = [
            create_row_rates(row_ind, row_ind in absorbing_states, ser_rates[row_ind], non_abrosing_out_rates, ph_size,
                             non_absorbing) for row_ind in range(ph_size)]
        A = np.concatenate(lists_rate_mat).reshape((ph_size, ph_size))  ## converting all into one numpy array

        num_of_pos_initial_states = np.random.randint(1, ph_size + 1)
        non_zero_probs = np.random.dirichlet(np.random.rand(num_of_pos_initial_states), 1)
        inds_of_not_zero_probs = np.sort(np.random.choice(ph_size, num_of_pos_initial_states, replace=False))
        s = np.zeros(ph_size)
        s[inds_of_not_zero_probs] = non_zero_probs

    else:
        s = np.array([1.])
        potential_vals = np.linspace(scale_low, scale_high, 20000)
        randinds = np.random.randint(potential_vals.shape[0], size=ph_size)
        ser_rates = (potential_vals[randinds]).reshape((1, ph_size))
        A = -ser_rates

    return (s, A)

def balance_sizes(sizes):
    for ind in range(sizes.shape[0]):
        if sizes[ind] < 3:
            ind_max = np.argmax(sizes)
            if sizes[ind_max] >2 :
                sizes[ind] +=1
                sizes[ind_max] -=1
    return sizes

def recursion_group_size(group_left, curr_vector, phases_left):
    if group_left == 1:
        return np.append(phases_left, curr_vector)
    else:

        if phases_left + 1 - group_left == 1:
            curr_size = 1
        else:
            curr_size =  1+ np.random.binomial(phases_left + 1 - group_left-1, np.random.uniform(0.1,0.5))
        return recursion_group_size(group_left - 1, np.append(curr_size, curr_vector), phases_left - curr_size)

def create_mix_erlang_ph(ph_size, scale_low=1, max_scale_high=15, max_ph=500):

    if ph_size > 2:
        ph_size_gen_ph = np.random.randint(2, ph_size)

    else:
        return create_gen_erlang_many_ph(ph_size)

    erlang_max_size = np.random.randint(int(0.25 * max_ph), int(0.75 * max_ph))

    scale_high = np.random.uniform(2, max_scale_high)
    # ph_size_gen_ph = np.random.randint(5, max_ph - erlang_max_size)
    # if int(0.5*ph_size_gen_ph ) > 1:

    #     num_groups = np.random.randint(1,  int(0.5*ph_size_gen_ph) )
    # else:
    #     num_groups = 1
    num_groups = sample_num_groups(ph_size_gen_ph)


    # group_sizes = np.random.randint(1, 25, num_groups)

    group_sizes_gen_ph = recursion_group_size(num_groups, np.array([]), ph_size_gen_ph) #(group_sizes * ph_size_gen_ph / np.sum(group_sizes)).astype(int) + 1
    if np.random.rand()>0.01:
        group_sizes_gen_ph = balance_sizes(group_sizes_gen_ph)
    erlang_list_gen_ph = [give_s_A_given__fixed_size(size, scale_low, scale_high) for size in group_sizes_gen_ph.astype(int)]
    erlang_list_gen_ph_A = [lis[1] for lis in erlang_list_gen_ph]
    erlang_list_gen_ph_s = [lis[0] for lis in erlang_list_gen_ph]

    ph_size_erl = ph_size - ph_size_gen_ph #np.random.randint(5, erlang_max_size)
    # if ph_size_erl > 2:
    #     num_groups = np.random.randint(1, min(7, ph_size_erl - 1))
    # else:
    #     num_groups = 1
    num_groups = sample_num_groups(ph_size_erl)


    # group_sizes = recursion_group_size(num_groups, np.array([]), ph_size_erl).astype(int)  #np.random.randint(1, 25, num_groups)
    if np.random.rand() > 0.8:
        rates = np.random.rand(num_groups)*200   #((np.ones(num_groups) * np.random.uniform(1, 1.75)) ** np.arange(num_groups))
    else:
        rates = np.random.uniform(1, 1.75) ** (np.random.rand(num_groups) * 10)
    group_sizes_erl = recursion_group_size(num_groups, np.array([]), ph_size_erl).astype(int) # (group_sizes * ph_size_erl / np.sum(group_sizes)).astype(int) + 1
    if np.random.rand()>0.01:
        group_sizes_erl = balance_sizes(group_sizes_erl)
    erlang_list_erl = [generate_erlang_given_rates(rates[ind], ph_size_erl) for ind, ph_size_erl in
                       enumerate(group_sizes_erl)]
    group_sizes = np.append(group_sizes_gen_ph, group_sizes_erl)
    group_sizes = group_sizes.astype(int)
    rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)

    ph_list = erlang_list_gen_ph_A + erlang_list_erl

    ph_size = np.sum(group_sizes)
    A = np.zeros((int(ph_size), int(ph_size)))
    s = np.zeros(int(ph_size))
    for ind in range(group_sizes.shape[0]):
        if ind < group_sizes_gen_ph.shape[0]:
            s[int(np.sum(group_sizes[:ind])):int(np.sum(group_sizes[:ind]) + group_sizes[ind])] = rand_probs[0][ind] * \
                                                                                        erlang_list_gen_ph_s[ind]
        else:
            s[np.sum(group_sizes[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
        A[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
        np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = ph_list[ind]

    fst_mom = compute_first_n_moments(s, A, 1)
    if type(fst_mom[0]) != bool:
        A = A * fst_mom[0][0]
        fst_mom = compute_first_n_moments(s, A, 1)
        return (s, A)

    else:
        return False



def create_gen_erlang_many_ph(ph_size):
    # ph_size = np.random.randint(1, max_ph_size)

    num_groups = sample_num_groups(ph_size)


    # if ph_size > 1:
    #     num_groups = np.random.randint(2,min(8,ph_size))
    # else:
    #     num_groups = 1
    group_sizes_1 = recursion_group_size(num_groups, np.array([]), ph_size).astype(int)
    if np.random.rand()>0.01:
        group_sizes_1 = balance_sizes(group_sizes_1)
    rates = np.random.uniform(1, 1.75)**(np.random.rand(num_groups)*10) # ((np.ones(num_groups)*np.random.uniform(1, 1.85))**np.arange(num_groups))
    s,A = create_gen_erlang_given_sizes(group_sizes_1, rates)

    A = A*compute_first_n_moments(s, A, 1)[0][0]
    return (s,A)

def sample_num_groups(n, thresh =0.98):
    if np.random.rand()>thresh:
        num = 1+np.random.binomial(n-1, np.random.uniform(0.2,0.99))
    elif np.random.rand()>0.9:
        num = 1+np.random.binomial(int(n*0.1), np.random.uniform(0.3,0.87))
    else:
        if n<10:
            portion = 0.3
        else:
            portion = 0.8
        num = 1+np.random.binomial(min(10,int(n-1)*portion), np.random.uniform(0.1,0.9))
    if (num==1) & (n>1 ) &(np.random.rand()>0.4):
        num +=1
    return num

def create_Erlang_given_ph_size(ph_size):
    s = np.zeros(ph_size)
    s[0] = 1
    rate = ph_size
    A = generate_erlang_given_rates(rate, ph_size)
    # A = A*compute_first_n_moments(s, A, 1)[0][0]
    return (s,A)

def send_to_the_right_generator(num_ind, ph_size):

    if num_ind == 1: ## Any arbitrary ph
        s_A =  create_mix_erlang_ph(ph_size) # give_s_A_given_size(np.random.randint(60, max_ph_size))
    elif num_ind > 1:
        s_A = create_gen_erlang_many_ph(ph_size)
    else:
        s_A = create_Erlang_given_ph_size(ph_size)
    if type(s_A) != bool:
        try:

            s = s_A[0]
            A = s_A[1]

            return (s,A)

        except:
            print('Not able to extract s and A')


def gamma_pdf(x, theta, k):
    return (1 / (gamma(k))) * (1 / theta ** k) * (np.exp(-x / theta))


def gamma_lst(s, theta, k):
    return (1 + theta * s) ** (-k)

def gamma_mfg(shape, scale, s):
    return (1-scale*s)**(-shape)

def get_nth_moment(shape, scale, n):
    s = Symbol('s')
    y = gamma_mfg(shape, scale, s)
    for i in range(n):
        if i == 0:
            dx = diff(y,s)
        else:
            dx = diff(dx,s)
    return dx.subs(s, 0)


# def get_nth_moment(theta, k, n):
#     s = Symbol('s')
#     y = gamma_lst(s, theta, k)
#     for i in range(n):
#         if i == 0:
#             dx = diff(y, s)
#         else:
#             dx = diff(dx, s)
#     return ((-1) ** n) * dx.subs(s, 0)


def gamma_lst(s, theta, k):
    return (1 + theta * s) ** (-k)


def unif_lst(s, b, a=0):
    return (1 / (b - a)) * ((np.exp(-a * s) - np.exp(-b * s)) / s)


def n_mom_uniform(n, b, a=0):
    return (1 / ((n + 1) * (b - a))) * (b ** (n + 1) - a ** (n + 1))


def laplace_mgf(t, mu, b):
    return exp(mu * t) / (1 - (b ** 2) * (t ** 2))


def nthmomlap(mu, b, n):
    t = Symbol('t')
    y = laplace_mgf(t, mu, b)
    for i in range(n):
        if i == 0:
            dx = diff(y, t)
        else:
            dx = diff(dx, t)
    return dx.subs(t, 0)


def normal_mgf(t, mu, sig):
    return exp(mu * t + (sig ** 2) * (t ** 2) / 2)


def nthmomnormal(mu, sig, n):
    t = Symbol('t')
    y = normal_mgf(t, mu, sig)
    for i in range(n):
        if i == 0:
            dx = diff(y, t)
        else:
            dx = diff(dx, t)
    return dx.subs(t, 0)


def generate_unif(is_arrival):
    if is_arrival:
        b_arrive = np.random.uniform(2.1, 2.1)
        a_arrive = 0
        moms_arr = []
        for n in range(1, 11):
            moms_arr.append(n_mom_uniform(n, b_arrive))
        return (a_arrive, b_arrive, moms_arr)
    else:
        b_ser = 2
        a_ser = 0
        moms_ser = []
        for n in range(1, 11):
            moms_ser.append(n_mom_uniform(n, b_ser))
        return (a_ser, b_ser, moms_ser)




def generate_gamma(is_arrival):
    if is_arrival:
        rho = np.random.uniform(0.9, 0.99)
        shape = 1 # np.random.uniform(0.1, 100)
        scale = 1 / (rho * shape)
        moms_arr = np.array([])
        for mom in range(1, 11):
            moms_arr = np.append(moms_arr, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
        return (shape, scale, moms_arr)
    else:
        shape = np.random.uniform(1, 100)
        scale = 1 / shape
        moms_ser = np.array([])
        for mom in range(1, 11):
            moms_ser = np.append(moms_ser, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
        return (shape, scale, moms_ser)


def generate_normal(is_arrival):
    if is_arrival:
        mu = np.random.uniform(1.1, 1.2)
        sig = np.random.uniform(mu / 6, mu / 4)

        moms_arr = np.array([])
        for mom in tqdm(range(1, 11)):
            moms_arr = np.append(moms_arr, np.array(N(nthmomnormal(mu, sig, mom))).astype(np.float64))

        return (mu, sig, moms_arr)
    else:
        mu = 1
        sig = np.random.uniform(0.15, 0.22)

        moms_ser = np.array([])
        for mom in tqdm(range(1, 11)):
            moms_ser = np.append(moms_ser, np.array(N(nthmomnormal(mu, sig, mom))).astype(np.float64))
        return (mu, sig, moms_ser)


def gg1_generator( util_lower = 0.5, util_upper = 0.99, arrival_dist =  4, ser_dist = 4):
    '''
    Generate G/G/1 queue where is G is PH
    util_lower: lower bound of queue utilization
    util_upper: upper bound of queue utilization
    arrival_dist: give code of the distirubtion type - 4 means ph
    ser_dist: give code of the distirubtion type - 4 means ph
    '''

    from datetime import datetime

    now = datetime.now()
    np.random.seed(now.microsecond)
    rho = np.random.uniform(util_lower, util_upper)



    if np.random.rand() < 0.8:
        s_arrival, A_arrival = create_gen_erlang_many_ph(np.random.randint(5, 50))
    else:
        try:
            s_arrival, A_arrival = create_mix_erlang_ph(np.random.randint(5, 50))
        except:
            s_arrival, A_arrival = create_gen_erlang_many_ph(np.random.randint(5, 50))

    moms_arrive = np.array(compute_first_n_moments(s_arrival, A_arrival, 10)).flatten()

    arrival_rate = 1/moms_arrive[0]

    if np.random.rand() < 0.8:
        s_service, A_service = create_gen_erlang_many_ph(np.random.randint(35, 101))
    else:
        try:
            s_service, A_service = create_mix_erlang_ph(np.random.randint(35, 101))
        except:
            s_service, A_service = create_gen_erlang_many_ph(np.random.randint(35, 101))


    A_service = A_service/rho

    moms_service = np.array(compute_first_n_moments(s_service, A_service, 10)).flatten()

    rho = arrival_rate*moms_service[0]



    return (arrival_dist, ser_dist, (s_arrival, A_arrival, moms_arrive), (s_service, A_service, moms_service))

def generate_mgc(capacity, util_lower = 0.7, util_upper = 1.1, arrival_dist =  4, ser_dist = 4):
    '''
    Generate M/G/c queue where is G is PH
    util_lower: lower bound of queue utilization
    util_upper: upper bound of queue utilization
    arrival_dist: give code of the distirubtion type - 4 means ph
    ser_dist: give code of the distirubtion type - 4 means ph
    '''

    from datetime import datetime

    now = datetime.now()

    np.random.seed(now.microsecond)
    arrival_rate = np.random.uniform(0, 1)
    A_arrival = np.array([[-1.]])
    s_arrival = np.array([1.])
    A_arrival = A_arrival * arrival_rate

    moms_arrive = np.array(compute_first_n_moments(s_arrival, A_arrival, 10)).flatten()

    if np.random.rand() < 0.8:
        s_service, A_service = create_gen_erlang_many_ph(np.random.randint(35, 101))
    else:
        try:
            s_service, A_service = create_mix_erlang_ph(np.random.randint(35, 101))
        except:
            s_service, A_service = create_gen_erlang_many_ph(np.random.randint(35, 101))


    ex_lower = util_lower*capacity/arrival_rate
    ex_upper = util_upper*capacity/arrival_rate

    ex = np.random.uniform(ex_lower, ex_upper)

    A_service = A_service/ex

    moms_service = np.array(compute_first_n_moments(s_service, A_service, 10)).flatten()

    rho = arrival_rate*moms_service[0]/capacity

    return (arrival_dist, ser_dist, (s_arrival, A_arrival, moms_arrive), (s_service, A_service, moms_service))

def generate_mmc(capacity, util_lower=0.7, util_upper=1.1, arrival_dist=4, ser_dist=4):
    '''
    Generate M/M/c queue where is G is PH
    util_lower: lower bound of queue utilization
    util_upper: upper bound of queue utilization
    arrival_dist: give code of the distirubtion type - 4 means ph
    ser_dist: give code of the distirubtion type - 4 means ph
    '''

    from datetime import datetime

    now = datetime.now()

    np.random.seed(now.microsecond)
    rho = np.random.uniform(util_lower, util_upper)
    arrival_rate = rho * capacity
    A_arrival = np.array([[-1.]])
    s_arrival = np.array([1.])
    # A_arrival = A_arrival * arrival_rate

    moms_arrive = np.array(compute_first_n_moments(s_arrival, A_arrival, 10)).flatten()

    A_service = np.array([[-1.]])/rho
    s_service = np.array([1.])

    moms_service = np.array(compute_first_n_moments(s_service, A_service, 10)).flatten()

    rho = arrival_rate * moms_service[0] / capacity

    return (arrival_dist, ser_dist, (s_arrival, A_arrival, moms_arrive), (s_service, A_service, moms_service))

def generate_ph(is_arrival, is_exponential):

    if is_exponential:
        s1, A1 = create_gen_erlang_many_ph(1)

    if np.random.rand() < 0.8:
        s1, A1 = create_gen_erlang_many_ph(np.random.randint(35, 201))
    else:
        try:
            s1, A1 = create_mix_erlang_ph(np.random.randint(35, 201))
        except:
            s1, A1 = create_gen_erlang_many_ph(np.random.randint(35, 101))

    if is_arrival:
        rho  = np.random.uniform(0.75, 1)
        A1 = A1*rho

    moms_ser = np.array(compute_first_n_moments(s1, A1, 10)).flatten()

    return (s1, A1, moms_ser)


def find_num_cust_time_stamp(df, time):

    if time == 0:
        return df.loc[df['Time_stamp'] == 0, :].shape[0]

    if df.loc[df['Time_stamp'] < time, :].shape[0] == 0:
        return 0
    else:
        LB = df.loc[df['Time_stamp'] < time, :].index[-1]
        return df.loc[LB, 'num_cust']

class g:


    # capacity = 1
    inter_arrival = []
    service_times = []
    queueing_time = {}
    inter_departure_time = []
    num_arrivals = 450000
    warm_up_arrivals = 0
    num_moms = 10
    counter_for_moms_arrivals = 0
    counter_for_moms_depart_sojourn = 0

    end_time = 40
    max_num_customers = 500



class Customer:
    def __init__(self, p_id,  arrival_time):
        self.id = p_id
        self.arrival_time = arrival_time

class GG1:

    def __init__(self,  arrive_means_per_group, ser_means_per_group, arrivals_dicts,  services_dicts,
                                    all_arrival_means, ser_mean, cycle_size, num_cycles, service_groups, group_size_arrive,initial):

        self.arrive_means_per_group = arrive_means_per_group
        self.ser_means_per_group = ser_means_per_group
        self.arrivals_dicts = arrivals_dicts
        self.services_dicts = services_dicts
        self.all_arrival_means = all_arrival_means
        self.ser_mean = ser_mean
        self.cycle_size = cycle_size
        self.num_cycles = num_cycles
        self.service_groups = service_groups
        self.group_size_arrive = group_size_arrive
        self.initial = initial

        self.customer_counter = 0
        self.env = simpy.Environment()
        self.patient_counter = 0

        self.server = simpy.PriorityResource(self.env, capacity=1)


        self.last_event_time = 0
        self.last_time = 0
        self.last_departure = 0
        self.prev_arrival = 0
        self.prev_departure = 0
        self.event_log_customer_id_list = []
        self.event_log_num_cust_list = []
        self.event_log_type_list = []
        self.event_log_time_stamp = []
        self.size_initial = 100
        self.num_cust_sys = 0



        self.initial_probs = initial


        self.event_log = pd.DataFrame([], columns = ['Customer_id',  'time_stamp', 'num_cust', 'event_type'])
        self.queueing_time = {}
        self.last1000 = time.time()

        self.sim_lenght_indicator = np.random.choice(3, 1, p=[0.0, 0.0, 1.0])[0]

        if self.sim_lenght_indicator == 0:
            self.end_time = 10
        elif self.sim_lenght_indicator == 1:
            self.end_time = 30
        else:
            self.end_time = 40

    def run(self):

        self.env.process(self.customer_arrivals())

        self.env.run(until=g.end_time)

    def give_group_per_time(self, groups, t):
        num_groups = groups.shape[0]
        a = np.tile(groups, self.num_cycles + 1)
        group_ind = np.where(a.cumsum() > t)[0][0]
        within_cycle = group_ind % num_groups
        return within_cycle

    def service(self, customer):

       arrival_time = self.env.now


       with self.server.request(priority=1) as req: #priority=priority
            yield req

            q_time = self.env.now - arrival_time

            # service time

            time_period = int(self.env.now)

            curr_group = self.give_group_per_time(self.service_groups, time_period)

            service_time = self.services_dicts[curr_group][3][self.customer_counter]

            yield self.env.timeout(service_time)

            self.prev_departure = self.last_departure
            self.last_departure = self.env.now
            tot_time = self.env.now - self.last_event_time
            self.num_cust_sys -= 1
            self.last_event_time = self.env.now
            self.last_time = self.env.now
            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)
            self.event_log_num_cust_list.append(self.num_cust_sys)
            self.event_log_type_list.append('Departure')


    def customer_arrivals(self):

        num_cust_init = np.random.choice(self.size_initial, 1, p=self.initial_probs)[0]

        for ind in range(num_cust_init):
            yield self.env.timeout(0)
            arrival_time = self.env.now
            customer = Customer(self.customer_counter, arrival_time)
            self.customer_counter += 1

            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)
            self.num_cust_sys += 1
            self.event_log_num_cust_list.append(self.num_cust_sys)

            self.event_log_type_list.append('Arrival')

            tot_time = self.env.now - self.last_event_time
            self.last_event_time = self.env.now

            self.last_time = self.env.now
            self.env.process(self.service(customer))

        while True:

            time_period = int(self.env.now)

            curr_group = self.give_group_per_time(self.group_size_arrive, time_period)

            inter_arrival_time = self.arrivals_dicts[curr_group][3][self.customer_counter]

            # print('The time is {} and the arrival rate {}, customer counter is {}.' .format(self.env.now, inter_arrival_rate,  self.customer_counter))
            yield self.env.timeout(inter_arrival_time)

            arrival_time = self.env.now
            customer = Customer(self.customer_counter, arrival_time)
            self.customer_counter += 1
            self.event_log_customer_id_list.append(customer.id)
            self.event_log_time_stamp.append(self.env.now)

            self.last_event_time = self.env.now
            self.num_cust_sys += 1
            self.event_log_num_cust_list.append(self.num_cust_sys)
            self.event_log_type_list.append('Arrival')
            self.last_time = self.env.now
            self.env.process(self.service(customer))

def single_sim(arrive_means_per_group, ser_means_per_group, arrivals_dicts,  services_dicts,
                                    all_arrival_means, ser_mean, cycle_size, num_cycles, service_groups, group_size_arrive, initial,  args):



    now = datetime.now()
    np.random.seed(now.microsecond)

    for ind in arrivals_dicts.keys():
        np.random.shuffle(arrivals_dicts[ind][3])

    for ind in services_dicts.keys():
        np.random.shuffle(arrivals_dicts[ind][3])


    gg1 = GG1(arrive_means_per_group, ser_means_per_group, arrivals_dicts,  services_dicts,
                                    all_arrival_means, ser_mean, cycle_size, num_cycles,service_groups, group_size_arrive, initial)
    gg1.run()
    # print("--- %s seconds the %d th iteration ---" % (time.time() - start_time, ind))

    gg1.event_log = pd.DataFrame({'Customer_id': gg1.event_log_customer_id_list,
                                  'Time_stamp': gg1.event_log_time_stamp,
                                  'Type': gg1.event_log_type_list,
                                  'num_cust': gg1.event_log_num_cust_list})

    result = [(time_epoch, find_num_cust_time_stamp(gg1.event_log, time_epoch)) for time_epoch in range(gg1.end_time)]
    resultDictionary = dict((x, y) for x, y in result)
    return resultDictionary





def give_service_mean_within_cycle(cycle_size):
    # num_groups = min(np.random.geometric(p=0.5),cycle_size)
    num_groups, group_sizes = give_group_size(
        cycle_size)  # recursion_group_size(num_groups, np.array([]), cycle_size).astype(int)
    service_mean = np.random.randint(1, 2, num_groups) #np.random.uniform(0.5, 3, num_groups)
    service_mean = service_mean.astype(float)
    services_arr = np.array([])
    for ind, ser_mean in enumerate(service_mean):
        services_arr = np.concatenate((services_arr, np.ones(group_sizes[ind]) * ser_mean))

    return (services_arr, group_sizes)





def give_rhos(pick_rho, num_pirets, is_increasing):
    # num_groups = min(np.random.geometric(p=0.35), num_pirets)
    # # print('num pirets {} num groups {}' .format(num_pirets, num_groups))
    # group_sizes = recursion_group_size(num_groups, np.array([]), num_pirets).astype(int)

    num_groups, group_sizes = give_group_size(num_pirets)

    if is_increasing:
        rates = np.sort(np.random.uniform(0.5, pick_rho, num_groups))
    else:
        rates = np.flip(np.sort(np.random.uniform(pick_rho, 5, num_groups)))

    rates_arr = np.array([])
    for ind, rate in enumerate(rates):
        rates_arr = np.concatenate((rates_arr, np.ones(group_sizes[ind]) * rate))

    return (rates_arr, group_sizes)


def give_service_mean_within_cycle(cycle_size ):
    # num_groups = min(np.random.geometric(p=0.5),cycle_size)
    num_groups, group_sizes = give_group_size(
        cycle_size)  # recursion_group_size(num_groups, np.array([]), cycle_size).astype(int)
    service_mean = np.random.uniform(0.5, 3, num_groups)
    services_arr = np.array([])
    for ind, ser_mean in enumerate(service_mean):
        services_arr = np.concatenate((services_arr, np.ones(group_sizes[ind]) * ser_mean))

    return (services_arr, group_sizes)


def give_service_mean_within_cycle(cycle_size ):
    # num_groups = min(np.random.geometric(p=0.5),cycle_size)
    num_groups, group_sizes = 1, np.array([cycle_size])
    service_mean = np.random.randint(1, 2, num_groups) #np.random.uniform(0.5, 3, num_groups)
    services_arr = np.array([])
    for ind, ser_mean in enumerate(service_mean):
        services_arr = np.concatenate((services_arr, np.ones(group_sizes[ind]) * ser_mean))

    return (services_arr, group_sizes)

def give_rhos_arrival_rates_ser_mean(vector_lenght=80):
    cycle_size = np.random.randint(10, 30)
    num_cycles = int(vector_lenght / cycle_size)

    last_cycle = vector_lenght - cycle_size * num_cycles
    # print('cycle size: {}, num cycles: {}, sum full cycles: {}, last cycle: {}.' .format(cycle_size, num_cycles, cycle_size*num_cycles, last_cycle))

    avg_rho = np.random.uniform(0.7, 0.99)
    pick = np.random.randint(3, cycle_size - 3)

    if np.random.rand() < 0.6:
        pick_rho = np.random.uniform(2, 30, 1)
    else:
        pick_rho = np.random.uniform(70, 100, 1)

    if pick_rho < 1.2:
        first_is_low = True
    else:
        first_is_low = False

    if first_is_low:
        first_batch, rhos_groups_1 = give_rhos(pick_rho, pick - 1, True)
        second_batch, rhos_groups_2 = give_rhos(pick_rho, cycle_size - pick, False)
    else:
        first_batch, rhos_groups_1 = give_rhos(pick_rho, pick - 1, False)
        second_batch, rhos_groups_2 = give_rhos(pick_rho, cycle_size - pick, True)

    rhos_groups = np.concatenate((rhos_groups_1, np.array([1]), rhos_groups_2), axis=0)
    rhos = np.concatenate((first_batch, pick_rho, second_batch), axis=0)
    mean_rhos = rhos.mean()
    rhos = (rhos / mean_rhos) * avg_rho
    all_rhos = np.tile(rhos, num_cycles + 1)[:vector_lenght]

    ser_mean, service_groups = give_service_mean_within_cycle(cycle_size)
    all_ser = np.tile(ser_mean, num_cycles + 1)[:vector_lenght]

    all_rates = all_rhos / all_ser

    all_arrival_means = 1 / all_rates

    unique_rates = np.array(list(set(all_rates[:cycle_size])))
    all_rates[:cycle_size][all_rates[:cycle_size] == unique_rates[0]].shape
    group_size_arrive = []
    for rate in unique_rates:
        group_size_arrive.append(all_rates[:cycle_size][all_rates[:cycle_size] == rate].shape[0])

    return (all_rhos, all_arrival_means, all_ser, service_groups, np.array(group_size_arrive), rhos_groups, cycle_size,
            num_cycles)

def give_rhos_arrival_rates_ser_mean(vector_lenght=80):

    cycle_size = np.random.randint(10, 40)
    num_cycles = int(vector_lenght / cycle_size)

    last_cycle = vector_lenght - cycle_size * num_cycles
    # print('cycle size: {}, num cycles: {}, sum full cycles: {}, last cycle: {}.' .format(cycle_size, num_cycles, cycle_size*num_cycles, last_cycle))

    avg_rho = np.random.uniform(0.8, 0.97)
    # pick = np.random.randint(3, cycle_size - 3)
    #
    # if np.random.rand() < 0.6:
    #     pick_rho = np.random.uniform(2, 30, 1)
    # else:
    #     pick_rho = np.random.uniform(70, 100, 1)
    #
    # if pick_rho < 1.2:
    #     first_is_low = True
    # else:
    #     first_is_low = False
    #
    # if first_is_low:
    #     first_batch, rhos_groups_1 = give_rhos(pick_rho, pick - 1, True)
    #     second_batch, rhos_groups_2 = give_rhos(pick_rho, cycle_size - pick, False)
    # else:
    #     first_batch, rhos_groups_1 = give_rhos(pick_rho, pick - 1, False)
    #     second_batch, rhos_groups_2 = give_rhos(pick_rho, cycle_size - pick, True)

    rhos_groups =  np.array([cycle_size])# np.concatenate((rhos_groups_1, np.array([1]), rhos_groups_2), axis=0)
    # rhos = np.concatenate((first_batch, pick_rho, second_batch), axis=0)
    rhos = np.ones(cycle_size)*avg_rho


    all_rhos = np.tile(rhos, num_cycles + 1)[:vector_lenght]

    ser_mean, service_groups = give_service_mean_within_cycle(cycle_size)
    all_ser = np.tile(ser_mean, num_cycles + 1)[:vector_lenght]

    all_rates = all_rhos / all_ser

    all_arrival_means = 1 / all_rates

    unique_rates = np.array(list(set(all_rates[:cycle_size])))
    all_rates[:cycle_size][all_rates[:cycle_size] == unique_rates[0]].shape
    group_size_arrive = []
    for rate in unique_rates:
        group_size_arrive.append(all_rates[:cycle_size][all_rates[:cycle_size] == rate].shape[0])

    return (all_rhos, all_arrival_means, all_ser, service_groups, np.array(group_size_arrive), rhos_groups, cycle_size,
            num_cycles)



def give_group_size(phases):
    num_groups = 1 + min(np.random.geometric(p=np.random.uniform(0.5, 0.7)), int(phases / 5))
    group_size = np.ones(num_groups) + \
                 np.random.multinomial(phases - num_groups, [1 / num_groups] * num_groups, size=1)[0]
    return num_groups, group_size.astype(int)



def create_dist_dicts(path, mean_per_group, group_sizes):
    num_groups = group_sizes.shape[0]
    realization_files = os.listdir(path)
    inds_realizations = np.random.choice(len(realization_files), num_groups, replace=False)
    dist_dicts = {}
    for ind, _ in enumerate(inds_realizations):
        dist = pkl.load(open(os.path.join(path, realization_files[inds_realizations[ind]]), 'rb'))
        dist_dicts[ind] = dist
        new_realizations = dist_dicts[ind][3] * mean_per_group[ind]
        new_A = dist_dicts[ind][1] / mean_per_group[ind]
        new_moms = np.array(compute_first_n_moments(dist_dicts[ind][0], new_A, n=10)).flatten()
        dist_dicts[ind] = (dist_dicts[ind][0], new_A, new_moms, new_realizations)

    return dist_dicts


def create_mean_per_group_dict(all_means):
    df = pd.DataFrame(all_means, columns=['means'])
    means_per_group = {}
    for group, mean in enumerate(df['means'].unique()):
        means_per_group[group] = mean
    return means_per_group



def run_single_setting(args):

    # s_service, A_service, moms_service = model_inputs

    now = datetime.now()

    if 'dkrass' in os.getcwd().split('/'):
        dists_path = '/scratch/d/dkrass/eliransc/services'
    elif 'C:' in os.getcwd().split('/')[0]:
        dists_path = r'C:\Users\user\workspace\data\ph_random\ph_mean_1_one_per_pkl'
    else:
        dists_path = '/scratch/eliransc/ph_random/large_ph_one_in_pkl_mdium/' # large_ph_one_in_pkl'   #

    all_rhos, all_arrival_means, ser_mean, service_groups, group_size_arrive, rhos_groups, cycle_size, num_cycles\
        = give_rhos_arrival_rates_ser_mean(args.number_sequences)

    arrive_means_per_group = create_mean_per_group_dict(all_arrival_means)
    ser_means_per_group = create_mean_per_group_dict(ser_mean)

    arrivals_dicts = create_dist_dicts(dists_path, arrive_means_per_group, group_size_arrive)
    services_dicts = create_dist_dicts(dists_path, ser_means_per_group, service_groups)

    np.random.seed(now.microsecond)

    size_initial = 100
    num_positive_values = np.random.randint(5, 15)
    s = np.random.dirichlet(np.ones(num_positive_values))
    initial = np.concatenate((s, np.zeros(size_initial - num_positive_values)))

    time_dict = {}
    for time_ in range(g.end_time):
        time_dict[time_] = np.zeros(g.max_num_customers)

    model_num = np.random.randint(1, 100000000)

    list_of_lists1 = []
    for ind in tqdm(range(10)):
        list_of_dicts = [single_sim(arrive_means_per_group, ser_means_per_group, arrivals_dicts,  services_dicts,
                                    all_arrival_means, ser_mean, cycle_size, num_cycles,service_groups, group_size_arrive, initial,  args) for ind in
                          range(1, args.num_iter_same_params + 1)]
        list_of_lists1.append(list_of_dicts)

    merged1 = list(itertools.chain(*list_of_lists1))

    for resultDictionary in merged1:
        for time1 in resultDictionary.keys():
            time_dict[time1][resultDictionary[time1]] += 1

    curr_path1 ='Model_number_' + str(model_num) + '_sequence_length_' + str(args.number_sequences) + '.pkl'

    full_path1 = os.path.join(args.read_path, curr_path1)

    row_cycle_info = (all_rhos, all_arrival_means, ser_mean, service_groups, group_size_arrive, rhos_groups, cycle_size, num_cycles)

    proecessed_cycle_data = (arrive_means_per_group, ser_means_per_group, arrivals_dicts, services_dicts)

    # pkl.dump((time_dict, row_cycle_info, proecessed_cycle_data, initial), open(full_path1, 'wb'))

    res_input, prob_queue_arr = create_single_data_point(args.read_path, curr_path1, time_dict, row_cycle_info, proecessed_cycle_data, initial)

    return res_input, prob_queue_arr


def create_single_data_point(path, file_name, time_dict, row_cycle_info, proecessed_cycle_data, initial, max_prob = 70):

    full_path = os.path.join(path, file_name)

    # time_dict, row_cycle_info, proecessed_cycle_data, initial = pkl.load(open(full_path, 'rb'))

    all_rhos, all_arrival_means, ser_mean, service_groups, group_size_arrive, rhos_groups, cycle_size, num_cycle = row_cycle_info

    arrive_means_per_group, ser_means_per_group, arrivals_dicts, services_dicts = proecessed_cycle_data

    res_input = np.array([])
    prob_queue_arr = np.array([])

    for t in range(len(time_dict)):

        arrive_group = give_group_per_time(group_size_arrive, t, num_cycle)
        service_group = give_group_per_time(service_groups, t, num_cycle)

        arr_input = np.concatenate((np.log(arrivals_dicts[arrive_group][2]), np.log(services_dicts[service_group][2]),
                                    np.array([t]), initial[:max_prob]), axis=0)
        arr_input = arr_input.reshape(1, arr_input.shape[0])
        probs = (time_dict[t] / time_dict[t].sum())
        max_or_more = probs[max_prob:].sum()
        probs_output = np.concatenate((probs[:max_prob], np.array([max_or_more])), axis=0)
        if t == 0:
            res_input = arr_input

            max_or_more = initial[max_prob:].sum()
            probs_output = np.concatenate((initial[:max_prob], np.array([max_or_more])), axis=0)

            prob_queue_arr = np.array(probs_output).reshape(1, probs_output.shape[0])
        else:
            res_input = np.concatenate((res_input, arr_input), axis=0)
            prob_queue_arr = np.concatenate((prob_queue_arr, np.array(probs_output).reshape(1, probs_output.shape[0])),
                                            axis=0)

    return (res_input, prob_queue_arr)


def give_group_per_time(groups, t, num_cycles):
    num_groups = groups.shape[0]
    a = np.tile(groups, num_cycles + 1)
    group_ind = np.where(a.cumsum() > t)[0][0]
    within_cycle = group_ind % num_groups
    return within_cycle


def dump_batch_data(curr_batch_data, args):
    input_result_all = np.array([])
    output_result_all = np.array([])

    for ind in range(len(curr_batch_data)):

        res_input, prob_queue_arr = curr_batch_data[ind]

        single_input = res_input.reshape(1, res_input.shape[0], res_input.shape[1])
        single_output = prob_queue_arr.reshape(1, prob_queue_arr.shape[0], prob_queue_arr.shape[1])
        if input_result_all.shape[0] == 0:
            input_result_all = single_input
            output_result_all = single_output
        else:
            input_result_all = np.concatenate((input_result_all, single_input), axis=0)
            output_result_all = np.concatenate((output_result_all, single_output), axis=0)

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    np.random.seed(now.microsecond)
    model_name = np.random.randint(0, 100000000)

    curr_path1 ='Model_number_' + str(model_name) + '_sequence_length_'\
                + str(args.number_sequences) + '_batch_size_' + str(args.batch_size) + '.pkl'

    full_path = os.path.join(args.batch_finalized_data_path, curr_path1)

    pkl.dump((input_result_all, output_result_all), open(full_path, 'wb'))

    # return input_result_all, output_result_all


def main(args):

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    np.random.seed(now.microsecond)

    if 'dkrass' in os.getcwd().split('/'):
        args.read_path = '/scratch/d/dkrass/eliransc/time_dependant_cyclic'
    elif 'C:' in os.getcwd().split('/')[0]:
        args.read_path = r'C:\Users\user\workspace\data\time_dependant_100'
        args.batch_finalized_data_path = r'C:\Users\user\workspace\data\time_dependant_100'
    else:
        args.read_path = '/scratch/eliransc/sim_trans_gt_gt_1'
        args.batch_finalized_data_path = '/scratch/eliransc/new_g_g_1_trans'

    for ind_num_batches in range(args.num_batches):

        curr_batch_data = []

        for ind in tqdm(range(args.batch_size)):

            start = time.time()
            ####################
            curr_batch_data.append(run_single_setting(args))
            ####################
            runtime = time.time() - start

        # pkl.dump(curr_batch_data, open(r'C:\Users\user\workspace\notebooks\time_dependant\batch_data.pkl', 'wb'))

        dump_batch_data(curr_batch_data, args)

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_batches', type=int, help='num batches in one run', default=800)
    parser.add_argument('--number_sequences', type=int, help='num sequences in a single sim', default=40)
    parser.add_argument('--max_capacity', type=int, help='maximum server capacity', default=1)
    parser.add_argument('--num_iter_same_params', type=int, help='nu, replications within same input', default=1000)
    parser.add_argument('--max_num_classes', type=int, help='max num priority classes', default=1)
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=1)
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=1000)
    parser.add_argument('--num_arrival', type=float, help='The number of total arrivals', default=100500)
    parser.add_argument('--batch_size', type=float, help='service rate of mismatched customers', default=16)
    parser.add_argument('--batch_finalized_data_path', type=str, help='service rate of mismatched customers', default='/scratch/eliransc/time_dependant_cyclic')
    parser.add_argument('--read_path', type=str, help='the path of the files to read from', default=  '/scratch/eliransc/time_dependant_cyclic' ) # r'C:\Users\user\workspace\data\time_dependant'
    parser.add_argument('--read_path_niagara', type=str, help='the path of the files to read from',
                        default='/scratch/d/dkrass/eliransc/time_dependant_cyclic')
    parser.add_argument('--dump_path', type=str, help='path to pkl folder', default= r'C:\Users\user\workspace\data\gg1_inverse_pkls' ) # '/scratch/eliransc/gg1_inverse_pkls'
    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

