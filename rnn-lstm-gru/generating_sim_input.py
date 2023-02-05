import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(r'G:\My Drive\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')

import os

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
from butools.fitting import *
from datetime import datetime
from fastbook import *
import itertools
from scipy.special import factorial

import pickle as pkl

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
from butools.fitting import *
from datetime import datetime
from fastbook import *
import itertools
from scipy.special import factorial

import pickle as pkl





def thresh_func(row):
    if (row['First_moment'] < mom_1_thresh) and (row['Second_moment'] < mom_2_thresh) and (
            row['Third_moment'] < mom_3_thresh):
        return True
    else:
        return False


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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def give_diff(v, i, ph_size):
    if i == 0:
        return v[i]
    elif i == len(v):
        return ph_size - v[-1]
    else:
        return v[i] - v[i - 1]


def generate_erlang(ind, ph_size, group):
    #     group = sequence[ind]

    A = np.identity(ph_size)
    s = np.zeros(ph_size)
    s[0] = 1
    #     ser_group_score = np.random.rand()
    #     if ser_group_score <1/3:
    #         group = 1
    #     elif ser_group_score>2/3:
    #         group = 3
    #     else:
    #         group = 2
    #     ser_rates = loguniform.rvs(1e-1, 10e2, size=1)
    ser_rates = np.random.uniform(0.5, 20)
    if group == 1:
        ser_rates = random.uniform(0.01, 0.1)
    elif group == 2:
        ser_rates = random.uniform(0.1, 0.5)
    else:
        ser_rates = random.uniform(1.0, 1.5)

    #     A = -A*ser_rates

    A_list = [create_erlang_row(ser_rates, ind, ph_size) for ind in range(ph_size)]
    A = np.concatenate(A_list).reshape((ph_size, ph_size))
    return (s, A)


def create_erlang_row(rate, ind, size):
    aa = np.zeros(size)
    if ind < size - 1:
        aa[ind + 1] = rate
    return aa

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


def get_rand_ph_dist(ph_size):
    potential_vals = np.linspace(0.1, 1, 2000)
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
    absorbing_states, non_absorbing

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
    lists_rate_mat = [
        create_row_rates(row_ind, row_ind in absorbing_states, ser_rates[row_ind], non_abrosing_out_rates, ph_size,
                         non_absorbing) for row_ind in range(ph_size)]
    A = np.concatenate(lists_rate_mat).reshape((ph_size, ph_size))  ## converting all into one numpy array

    num_of_pos_initial_states = np.random.randint(1, ph_size + 1)
    non_zero_probs = np.random.dirichlet(np.random.rand(num_of_pos_initial_states), 1)
    inds_of_not_zero_probs = np.sort(np.random.choice(ph_size, num_of_pos_initial_states, replace=False))
    s = np.zeros(ph_size)
    s[inds_of_not_zero_probs] = non_zero_probs

    return s, A


def compute_first_n_moments(s, A, n=3):
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list


def is_all_n_moments_defined(mom_list):
    for mom in mom_list:
        if not mom:
            return False
    else:
        return True


def increase_size_of_matrix(s, A, max_ph_size):
    ph_size = A.shape[0]
    new_A = np.zeros((max_ph_size, max_ph_size))
    new_s = np.zeros(max_ph_size)
    new_A[:ph_size, :ph_size] = A
    new_s[:ph_size] = s
    new_A[(np.arange(ph_size, max_ph_size), np.arange(ph_size, max_ph_size))] = -1
    return new_s, new_A


def thresh_func(row):
    if (row['First_moment'] < mom_1_thresh) and (row['Second_moment'] < mom_2_thresh) and (
            row['Third_moment'] < mom_3_thresh):
        return True
    else:
        return False


# def create_gen_erlang(sequenc, max_ph_size):
#     ph_size = np.random.randint(3, max_ph_size + 1)
#     num_samples_groups = min(ph_size, int(np.random.randint(2, ph_size) / 2))
#     v = np.sort(np.random.choice(ph_size, num_samples_groups, replace=False))
#     if v.shape[0] > 0:
#         diff_list = np.array([give_diff(v, i, ph_size) for i in range(len(v) + 1)])
#         diff_list = diff_list[diff_list > 0]
#
#         erlang_list = [generate_erlang(ind, ph_size, sequenc[ind]) for ind, ph_size in enumerate(diff_list)]
#         final_a = np.zeros((ph_size, ph_size))
#         final_s = np.zeros(ph_size)
#         rand_probs = np.random.dirichlet(np.random.rand(diff_list.shape[0]), 1)
#         for ind in range(diff_list.shape[0]):
#             final_s[np.sum(diff_list[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
#             final_a[np.sum(diff_list[:ind]):np.sum(diff_list[:ind]) + diff_list[ind],
#             np.sum(diff_list[:ind]):np.sum(diff_list[:ind]) + diff_list[ind]] = erlang_list[ind][1]
#         return increase_size_of_matrix(final_s, final_a, max_ph_size)
#     else:
#         print('Not valid')


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


# def create_gen_erlang_given_sizes(group_sizes, rates, ph_size, probs = False):
#     erlang_list = [generate_erlang_given_rates(rates[ind], ph_size) for ind, ph_size in enumerate(group_sizes)]
#     final_a = np.zeros((ph_size, ph_size))
#     final_s = np.zeros(ph_size)
#     if not probs.any():
#         rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)
#         rands = np.random.rand(group_sizes.shape[0])
#         rand_probs = rands / np.sum(rands).reshape((1, rand_probs.shape[0]))
#     else:
#         rand_probs = probs
#     for ind in range(group_sizes.shape[0]):
#         final_s[np.sum(group_sizes[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
#         final_a[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
#         np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = erlang_list[ind]
#     return final_s, final_a


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


def combine_erlangs_lists(data_path, pkl_name, UB_ratios_limits, ph_size_max, UB_rates=1, LB_rates=0.1,
                          num_examples_each_settings=500):
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S") + str(np.random.randint(1, 1000000, 1)[0]) + '.pkl'
    pkl_name = pkl_name + current_time

    pkl_full_path = os.path.join(data_path, pkl_name)

    UB_ratios = np.random.randint(UB_ratios_limits[0], UB_ratios_limits[1])
    UB_rates = 1
    LB_rates = 0.1
    num_groups_max = int(ph_size_max / 2)

    x = generate_mix_ph(ph_size_max, np.random.randint(max(3, int(UB_ratios / 2)), max(4, UB_ratios)),
                        np.random.uniform(UB_rates, 2 * UB_rates), LB_rates,
                        np.random.randint(3, num_groups_max + 1))

    y_data = compute_y_data_given_folder(x, max_ph_size, tot_prob=70)

    x_y_data = (x, y_data)

    pkl.dump(x_y_data, open(pkl_full_path, 'wb'))

    return x_y_data


def generate_mix_ph(ph_size_max, UB_ratios, UB_rates, LB_rates, num_groups_max):
    num_groups = np.random.randint(2, num_groups_max + 1)
    ph_size = np.random.randint(num_groups, ph_size_max + 1)
    #     lam_arr = np.zeros((max_ph_size+1,1))

    group_sizes = recursion_group_size(num_groups, np.array([]), ph_size).astype(int)

    ratios = np.random.randint(1, UB_ratios, num_groups - 1)
    ratios = np.append(1, ratios)
    first_rate = np.random.uniform(LB_rates, UB_rates)
    rates = first_rate * ratios

    gen_erlang = create_gen_erlang_given_sizes(group_sizes, rates)

    A, s = normalize_matrix(gen_erlang[0], gen_erlang[1])

    s, A = increase_size_of_matrix(s, A, ph_size_max)

    if compute_cdf(1, s, A) < 0.999:
        return generate_mix_ph(ph_size_max, UB_ratios, UB_rates, LB_rates, num_groups_max)

    final_data = create_final_x_data(s, A, ph_size_max)

    return final_data



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


def create_gewn_ph(ph_size_max, pkl_name, data_path):
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S") + str(np.random.randint(1, 1000000, 1)[0]) + '.pkl'
    pkl_name = pkl_name + current_time

    A_s_lists = [give_A_s_given_size(np.random.randint(2, ph_size_max)) for ind in range(1)]
    mom_lists = [compute_first_n_moments(tupl[1], tupl[0]) for tupl in A_s_lists]
    valid_list = [index for index, curr_mom_list in enumerate(mom_lists) if is_all_n_moments_defined(curr_mom_list)]
    normmat_ph_1 = [normalize_matrix(A_s[1], A_s[0]) for ind_A_s, A_s in enumerate(A_s_lists) if ind_A_s in valid_list]
    normmat_ph_1a = [(A_s[0], A_s[1]) for ind_A_s, A_s in enumerate(normmat_ph_1) if
                     compute_cdf(1, A_s[1], A_s[0]).flatten()[0] > 0.999]
    max_size_ph_1 = [increase_size_of_matrix(ph_dist[1], ph_dist[0], ph_size_max) for ph_dist in normmat_ph_1a]
    fin_data_reg = [create_final_x_data(ph_dist[0], ph_dist[1], ph_size_max) for ph_dist in max_size_ph_1]
    if len(fin_data_reg) > 0:
        x_y_data = compute_y_data_given_folder(fin_data_reg[0], ph_size_max, tot_prob=70)
        if type(x_y_data) == np.ndarray:
            pkl_full_path = os.path.join(data_path, pkl_name)
            pkl.dump((fin_data_reg[0], x_y_data), open(pkl_full_path, 'wb'))

            return (fin_data_reg[0], x_y_data)



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

            print(1-np.sum(steady_state), rho)
            steady_state = np.append(steady_state, 1-np.sum(steady_state))
            return steady_state
            # if np.sum(steady_state) > 1 - eps:
            #     return steady_state
            # else:
            #     return False

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


import random


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


# def create_gen_erlang_given_sizes(group_sizes, rates, probs=False):
#     ph_size = np.sum(group_sizes)
#     erlang_list = [generate_erlang_given_rates(rates[ind], ph_size) for ind, ph_size in enumerate(group_sizes)]
#     final_a = np.zeros((ph_size, ph_size))
#     final_s = np.zeros(ph_size)
#     if type(probs) == bool:
#         rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)
#         rands = np.random.rand(group_sizes.shape[0])
#         rand_probs = rands / np.sum(rands).reshape((1, rand_probs.shape[0]))
#     else:
#         rand_probs = probs
#     for ind in range(group_sizes.shape[0]):
#         final_s[np.sum(group_sizes[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
#         final_a[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
#         np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = erlang_list[ind]
#
#     return final_s, final_a


def find_upper_bound_rate_given_n(n, upper_bound):
    if upper_bound == 0:
        return False
    if np.array(get_lower_upper_x(n, upper_bound)).any():
        return find_upper_bound_rate_given_n(n, upper_bound - 1)
    else:
        return upper_bound + 1


def create_mix_erlang_data(max_num_groups=10):
    ph_sizes = np.linspace(10, 100, 10).astype(int)
    probs_ph_tot_size = np.array(ph_sizes ** 2 / np.sum(ph_sizes ** 2))
    num_groups = np.random.randint(1, max_num_groups + 1)
    group_size = recursion_group_size(num_groups, np.array([]),
                                      np.random.choice(ph_sizes, 1, p=probs_ph_tot_size)[0]).astype(int)
    group_rates = give_rates_given_Er_sizes(df_1, group_size, ratio_size)
    s, A = create_gen_erlang_given_sizes(group_size, group_rates)
    return (s, A)


def create_mix_erlang_data_steady(s, A, data_path, data_type, max_ph_size=100):
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S") + '_' + str(np.random.randint(1, 1000000, 1)[0]) + '.pkl'
    pkl_name = data_type + current_time

    pkl_full_path = os.path.join(data_path, pkl_name)

    A, s = normalize_matrix(s, A)

    s, A = increase_size_of_matrix(s, A, max_ph_size)

    final_data = create_final_x_data(s, A, max_ph_size)

    y_dat = compute_y_data_given_folder(final_data, max_ph_size, tot_prob=70)

    if type(y_dat) == bool:
        return False
        print('not dumping')
    else:
        pkl.dump((final_data, y_dat), open(pkl_full_path, 'wb'))
        return 1


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


def create_gen_erlang(UB_ratios=300, UB_rates=1, LB_rates=0.1,
                      ph_size_max=40, num_groups_max=10, ph_size_min=30):
    num_groups = np.random.randint(2, num_groups_max + 1)
    ph_size = np.random.randint(ph_size_min, ph_size_max + 1)
    group_sizes = recursion_group_size(num_groups, np.array([]), ph_size).astype(int)

    ratios = np.random.randint(1, UB_ratios, num_groups - 1)
    ratios = np.append(1, ratios)
    first_rate = np.random.uniform(LB_rates, UB_rates)
    rates = first_rate * ratios

    gen_erlang = create_gen_erlang_given_sizes(group_sizes, rates)
    curr_mean = ser_moment_n(gen_erlang[0], gen_erlang[1], 1)

    s = gen_erlang[0]
    A = gen_erlang[1]
    # norm_const = find_when_cdf_cross_0_999(s, gen_erlang[1], 0.5)
    # if norm_const == 0:
    #     print('Not able to find normalizing constant')
    #     return False
    # else:
     #A = gen_erlang[1] *norm_const

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
    num_groups = np.random.randint(2, min(20, ph_size_gen_ph - 1))
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

    if num_ind == 1: ## Any arbitrary ph
        s_A =  create_mix_erlang_ph() # give_s_A_given_size(np.random.randint(60, max_ph_size))
    elif num_ind > 1:
        s_A = create_gen_erlang_many_ph()
    else:
        s_A = create_Erlang_given_ph_size(max_ph_size)
    if type(s_A) != bool:
        try:
            # s_A = normalize_ph_so_it_1_when_cdf_1(s_A[0], s_A[1])
            # A = s_A[1]*compute_first_n_moments(s_A[0], s_A[1], 1)[0][0]
            s = s_A[0]
            A = s_A[1]

            return (s,A)

            # x = create_final_x_data(s, A, max_ph_size)
            # y = compute_y_data_given_folder(x, x.shape[0]-1, tot_prob=70, eps=0.0001)
            # if type(y) == np.ndarray:
            #     moms = compute_first_n_moments(s, A, num_moms)
            #
            #     mom_arr = np.concatenate(moms, axis=0)
            #
            #     lam = x[0, x.shape[0]-1]
            #     # mu = 1 / mom_arr[0]
            #     # lam = np.random.uniform(0 * mu, 0.95 * mu, 1)[0]
            #
            #     mom_arr = np.log(mom_arr)
            #     mom_arr = np.delete(mom_arr, 0)
            #     mom_arr = np.append(lam, mom_arr)
            #
            #     if not np.any(np.isinf(mom_arr)):
            #
            #         return (mom_arr, y)
        except:
            print('Not able to extract s and A')

def compute_y_moms(s,A,num_moms,max_ph_size):


    lam_vals = np.random.uniform(0.8, 0.99, 1)


    lam_y_list = []

    # mu = 1 / mom_arr[0]
    # lam = np.random.uniform(0 * mu, 0.95 * mu, 1)[0]
    for lam in lam_vals:
        x = create_final_x_data(s, A, lam)

        y = compute_y_data_given_folder(x, x.shape[0] - 1, tot_prob=500, eps=0.0001)
        if type(y) == np.ndarray:
            moms = compute_first_n_moments(s, A, num_moms)

            mom_arr = np.concatenate(moms, axis=0)

            lam = x[0, x.shape[0] - 1]
            # mu = 1 / mom_arr[0]
            # lam = np.random.uniform(0 * mu, 0.95 * mu, 1)[0]

            mom_arr = np.log(mom_arr)
            mom_arr = np.delete(mom_arr, 0)
            mom_arr = np.append(lam, mom_arr)

            if not np.any(np.isinf(mom_arr)):

                lam_y_list.append((mom_arr, y))

    return lam_y_list



def generate_one_ph(batch_size, max_ph_size, num_moms, data_path, data_sample_name):

    sample_type_arr = np.random.randint(1, 4, batch_size)
    x_y_moms_list = [send_to_the_right_generator(val, max_ph_size,  num_moms, data_path, data_sample_name) for val in sample_type_arr]
    x_y_moms_list = [x_y_moms for x_y_moms in x_y_moms_list if x_y_moms]
    x_y_moms_lists = [compute_y_moms(x_y_moms[0],x_y_moms[1], num_moms, max_ph_size) for x_y_moms  in x_y_moms_list]
    saving_batch(list(itertools.chain(*x_y_moms_lists)), data_path, data_sample_name, num_moms)

    ## Clean list

    # x_y_moms_list = [x_y_moms for x_y_moms in x_y_moms_list if x_y_moms]

    # for batch in range(8):
    #     x_y_moms_list = [send_to_the_right_generator(-1, batch*args.batch_size+ph_size,df_1, num_moms, data_path, data_sample_name) for ph_size in range(1,args.batch_size+1) if batch*args.batch_size+ph_size <=1000 ]
    #     x_y_moms_list = [x_y_moms for x_y_moms in x_y_moms_list if x_y_moms]
    #     saving_batch(x_y_moms_list, data_path, data_sample_name, num_moms)

    return 1

def create_Erlang_given_ph_size(ph_size):
    s = np.zeros(ph_size)
    s[0] = 1
    rate = ph_size
    A = generate_erlang_given_rates(rate, ph_size)
    # A = A*compute_first_n_moments(s, A, 1)[0][0]
    return (s,A)

def create_shrot_tale_genErlang(df_1, ratio_size=10):


    ph_sizes = np.linspace(10, 100, 10).astype(int)
    probs_ph_tot_size = np.array(ph_sizes ** 2 / np.sum(ph_sizes ** 2))
    num_groups = np.random.randint(1, 10)
    group_size = recursion_group_size(num_groups, np.array([]),
                                      np.random.choice(ph_sizes, 1, p=probs_ph_tot_size)[0]).astype(int)
    group_rates = give_rates_given_Er_sizes(df_1, group_size, ratio_size)
    s, A = create_gen_erlang_given_sizes(group_size, group_rates)

    return (s, A)


def generate_erlangs(batch_size, max_ph_size, num_moms, data_path, data_sample_name):

    sizes = np.random.randint(500,max_ph_size,batch_size)
    x_y_moms_list = [create_Erlang_given_ph_size(ph_size) for ph_size in sizes]
    x_y_moms_list = [x_y_moms for x_y_moms in x_y_moms_list if x_y_moms]
    x_y_moms_lists = [compute_y_moms(x_y_moms[0],x_y_moms[1], num_moms, max_ph_size) for x_y_moms  in x_y_moms_list]
    saving_batch(list(itertools.chain(*x_y_moms_lists)), data_path, data_sample_name, num_moms)

    ## Clean list

    # x_y_moms_list = [x_y_moms for x_y_moms in x_y_moms_list if x_y_moms]

    # for batch in range(8):
    #     x_y_moms_list = [send_to_the_right_generator(-1, batch*args.batch_size+ph_size,df_1, num_moms, data_path, data_sample_name) for ph_size in range(1,args.batch_size+1) if batch*args.batch_size+ph_size <=1000 ]
    #     x_y_moms_list = [x_y_moms for x_y_moms in x_y_moms_list if x_y_moms]
    #     saving_batch(x_y_moms_list, data_path, data_sample_name, num_moms)

    return 1


def is_dispatcher(id_case, df, ind_last_moved):
    arr = df.loc[df['id_case'] == id_case, :].index
    ind_moved = np.where(arr == ind_last_moved)[0][0]
    ind_potential_dispatcher = ind_moved - 1

    new_df = df.loc[df['id_case'] == id_case, :]
    new_df = new_df.reset_index()
    if new_df.loc[ind_potential_dispatcher, 'service_name'] == 'Dispatcher':
        return True
    else:
        return False


def dispatcher_start_time(id_case, df, ind_last_moved):
    arr = df.loc[df['id_case'] == id_case, :].index
    ind_moved = np.where(arr == ind_last_moved)[0][0]
    ind_potential_dispatcher = ind_moved - 1

    new_df = df.loc[df['id_case'] == id_case, :]
    new_df = new_df.reset_index()
    return new_df.loc[ind_potential_dispatcher, 'event_start_date']


def moved_start_time(id_case, df_, ind_last_moved):
    arr = df.loc[df_['id_case'] == id_case, :].index
    ind_moved = np.where(arr == ind_last_moved)[0][0]
    ind_potential_dispatcher = ind_moved - 1

    new_df = df_.loc[df_['id_case'] == id_case, :]
    new_df = new_df.reset_index()
    return new_df.loc[ind_moved, 'event_start_date']


def find_service_time(df, ind):
    return df.loc[ind, 'event_start_date']


def find_depart_time(df, ind):
    return df.loc[ind, 'event_end_date']


def moved_start_time(id_case, df_, ind_last_moved):
    arr = df_.loc[df_['id_case'] == id_case, :].index
    ind_moved = np.where(arr == ind_last_moved)[0][0]
    ind_potential_dispatcher = ind_moved - 1

    new_df = df_.loc[df_['id_case'] == id_case, :]
    new_df = new_df.reset_index()
    return new_df.loc[ind_moved, 'event_start_date']


def get_ind_of_the_last_moved(service_name, id_case, day, df):
    dd = df.loc[(df['service_name'] == service_name) & (df['event_name'] == 'Moved') & (df['id_case'] == id_case) & (
                df['day'] == day), ['index']].reset_index()

    if dd.shape[0] > 0:

        return np.array(dd['level_0'])[-1]
    else:
        return False


def check_call_after_moved(ind_moved, id_cust, df):
    for ind in range(ind_moved + 1, df.shape[0]):

        if (df.loc[ind, 'id_case'] == id_cust) & (df.loc[ind, 'event_name'] == 'Call'):
            return ind

    return 0


def compute_num_cust(df_ll):
    event_df = pd.DataFrame([])

    for ind in range(df_ll.shape[0]):
        curr_ind = event_df.shape[0]
        event_df.loc[curr_ind, 'event_type'] = 'Arrival'
        event_df.loc[curr_ind, 'time_stamp'] = df_ll.loc[ind, 'arrival_time']
        event_df.loc[curr_ind, 'case_id'] = df_ll.loc[ind, 'case_id']

        curr_ind = event_df.shape[0]
        event_df.loc[curr_ind, 'event_type'] = 'Departure'
        event_df.loc[curr_ind, 'time_stamp'] = df_ll.loc[ind, 'service_end']
        event_df.loc[curr_ind, 'case_id'] = df_ll.loc[ind, 'case_id']

    event_df = event_df.sort_values('time_stamp')
    event_df = event_df.reset_index()

    counter = 0
    for ind in range(event_df.shape[0]):
        if event_df.loc[ind, 'event_type'] == 'Arrival':
            counter += 1
        else:
            counter -= 1

        event_df.loc[ind, 'num_cust'] = counter

    without_date = pd.to_datetime(event_df['time_stamp']).apply(lambda d: d.time())
    event_df['time_stamp_without_date'] = without_date

    return event_df


def update_diff(df):
    df['hour_sero'] = '2015-01-01 08:00:00'

    t1 = pd.to_datetime(df['hour_sero'])
    t2 = pd.to_datetime(df['arrival_time'])
    df['tot_diff_arrival'] = t2 - t1

    df['tot_diff_arrival'] = df['tot_diff_arrival'].dt.total_seconds()

    t2 = pd.to_datetime(df['service_start'])
    df['tot_diff_service_start'] = t2 - t1

    df['tot_diff_service_start'] = df['tot_diff_service_start'].dt.total_seconds()

    t2 = pd.to_datetime(df['service_end'])
    df['tot_diff_service_end'] = t2 - t1

    df['tot_diff_service_end'] = df['tot_diff_service_end'].dt.total_seconds()

    return df


def give_num_at_end_epoch(df, time):
    if df.loc[df['time_stamp_without_date'] < time, :].shape[0] > 0:

        LB = df.loc[df['time_stamp_without_date'] < time, :].index[-1]

        return int(df.loc[LB, 'num_cust'])

    else:
        return 0



def main():
    event_dict = {}
    event_dict[1] = 'Enter the queue'
    event_dict[2] = 'Reception'
    event_dict[3] = 'Direct Reception'
    event_dict[4] = 'Schedule an appointment'
    event_dict[5] = 'Call next'
    event_dict[6] = 'Call'
    event_dict[7] = 'Call again'
    event_dict[8] = 'Unknown'
    event_dict[9] = 'Moved'
    event_dict[10] = 'Defrosted'
    event_dict[11] = 'Complete'
    event_dict[12] = 'Return to queue'
    event_dict[13] = 'Hold'
    event_dict[14] = 'Missing'
    event_dict[15] = 'Renewed'
    event_dict[16] = 'Release'
    event_dict[17] = 'Unknown2'
    event_dict[18] = 'Cancel enter the queue'
    event_dict[19] = 'Discontinued'
    event_dict[20] = 'Documentation'
    event_dict[23] = 'Unknown3'

    service_dict = {}
    service_dict[1] = 'EKG'
    service_dict[2] = 'Lab Tests'
    service_dict[3] = 'Gynecological Ultrasound'
    service_dict[4] = 'Gynecologist'
    service_dict[5] = 'Breast Surgeon'
    service_dict[6] = 'Cardiac Stress Test'
    service_dict[7] = 'Spirometry Exam'
    service_dict[8] = 'Ophthalmology Exam Technician'
    service_dict[9] = 'Ophthalmologist'
    service_dict[10] = 'Cardiologist'
    service_dict[11] = 'Physician'
    service_dict[12] = 'Hearing Test'
    service_dict[13] = 'Payment /forms'
    service_dict[14] = 'Nutritionist'
    service_dict[15] = 'Unknown1'
    service_dict[16] = 'Immunization adults'
    service_dict[17] = 'Unknown2'
    service_dict[18] = 'Unknown3'
    service_dict[19] = 'Research/ Urine Tests'
    service_dict[20] = 'Reception'
    service_dict[21] = 'End clinic service'
    service_dict[22] = 'Cardiac Stress Test finished'
    service_dict[23] = 'Unknown4'
    service_dict[24] = 'Dispatcher'
    service_dict[25] = 'External activities'
    service_dict[26] = 'Unknown5'
    service_dict[27] = 'Private / Family'
    service_dict[28] = 'Private / Family - Physician'
    service_dict[29] = 'Chest X-ray'
    service_dict[30] = 'Group'
    service_dict[31] = 'Unknown5'
    service_dict[32] = 'Aeronautical Physician'
    service_dict[33] = 'External activities'
    service_dict[34] = 'Occupational Physician'
    service_dict[35] = 'Flight Personnel Exam'
    service_dict[36] = 'Occupational Exam'

    df_15 = pd.read_excel('/home/eliransc/projects/def-dkrass/eliransc/RNN_queue/users_history_2015.xlsx')
    df_16 = pd.read_excel('/home/eliransc/projects/def-dkrass/eliransc/RNN_queue/users_history_2016.xlsx')

    service_name = 'Cardiologist'

    df_16['day'] = df_16.apply(lambda x: str(x.event_start_date.year) + '_' + str(x.event_start_date.month) + '_' + str(
        x.event_start_date.day), axis=1)
    df_16['service'].unique()
    df_16['event_name'] = df_16.apply(lambda x: event_dict[x.event], axis=1)
    df_16['service_name'] = df_16.apply(lambda x: service_dict[x.service], axis=1)
    days = df_16['day'].unique()
    df_16 = df_16.loc[(df_16['event_name'] != 'Schedule an appointment') & (df_16['event_name'] != 'Documentation'), :]
    df_16 = df_16.reset_index()

    df_15['day'] = df_15.apply(lambda x: str(x.event_start_date.year) + '_' + str(x.event_start_date.month) + '_' + str(
        x.event_start_date.day), axis=1)
    df_15['service'].unique()
    df_15['event_name'] = df_15.apply(lambda x: event_dict[x.event], axis=1)
    df_15['service_name'] = df_15.apply(lambda x: service_dict[x.service], axis=1)
    days = df_15['day'].unique()
    df_15 = df_15.loc[(df_15['event_name'] != 'Schedule an appointment') & (df_15['event_name'] != 'Documentation'), :]
    # df_15 = df_15.loc[~((df_15['event_name']=='Moved')&(df_15['service_name']=='Dispatcher')),:]
    df_15 = df_15.reset_index()

    df_list16 = []

    for day in tqdm(df_16['day'].unique()):

        event_list = []

        # day = '2015_1_20'
        df_ = df_16.loc[
            (df_16['day'] == day), ['id_case', 'service_name', 'event_name', 'location', 'day', 'start_date',
                                    'event_start_date', 'event_end_date']].sort_values(by=['event_start_date'])
        df_ = df_.reset_index()

        df_["start_date"] = df_["start_date"].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_["event_start_date"] = df_["event_start_date"].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_["event_end_date"] = df_["event_end_date"].dt.strftime('%Y-%m-%d %H:%M:%S')


        id_cases = df_.loc[df_['service_name'] == service_name, 'id_case'].unique()

        for id_case in id_cases:
            curr_inds = list((df_.loc[(df_['id_case'] == id_case) & (df_['service_name'] == service_name) & (
                        df_['day'] == day), 'event_name']).index)
            ind_last_moved = get_ind_of_the_last_moved(service_name, id_case, day, df_)
            event_name_list = list(df_.loc[(df_['id_case'] == id_case) & (df_['service_name'] == service_name) & (
                        df_['day'] == day), 'event_name'])
            if ind_last_moved:
                event_list.append((event_name_list, curr_inds, id_case, ind_last_moved))
            # print(df_.loc[(df_['id_case'] == id_case) & (df_['service_name'] == service_name) & (df_['day'] == day), ['event_name', 'start_date'	,'event_start_date'	,'event_end_date']])

        df_times = pd.DataFrame([])

        if len(event_list) > 0:
            for event in event_list:
                if len(event[0]) > 1:

                    service_start = find_service_time(df_, event[1][-1])
                    service_end = find_depart_time(df_, event[1][-1])

                    if is_dispatcher(event[2], df_, event[3]):

                        arrival_time = dispatcher_start_time(event[2], df_, event[3])

                        # print(arrival_time, service_start, service_end, 'dispatecher')
                    else:
                        arrival_time = moved_start_time(event[2], df_, event[3])
                        # print(arrival_time, service_start, service_end)

                    curr_ind = df_times.shape[0]
                    df_times.loc[curr_ind, 'case_id'] = event[2]
                    df_times.loc[curr_ind, 'arrival_time'] = arrival_time
                    df_times.loc[curr_ind, 'service_start'] = service_start
                    df_times.loc[curr_ind, 'service_end'] = service_end

                else:
                    pass

            df_list16.append(df_times.sort_values('arrival_time'))

    df_list = []

    for day in tqdm(df_15['day'].unique()):

        event_list = []

        # day = '2015_1_20'
        df_ = df_15.loc[
            (df_15['day'] == day), ['id_case', 'service_name', 'event_name', 'location', 'day', 'start_date',
                                    'event_start_date', 'event_end_date']].sort_values(by=['event_start_date'])
        df_ = df_.reset_index()

        df_["start_date"] = df_["start_date"].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_["event_start_date"] = df_["event_start_date"].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_["event_end_date"] = df_["event_end_date"].dt.strftime('%Y-%m-%d %H:%M:%S')
        # service_name = 'Ophthalmologist'

        id_cases = df_.loc[df_['service_name'] == service_name, 'id_case'].unique()

        for id_case in id_cases:
            curr_inds = list((df_.loc[(df_['id_case'] == id_case) & (df_['service_name'] == service_name) & (
                        df_['day'] == day), 'event_name']).index)
            ind_last_moved = get_ind_of_the_last_moved(service_name, id_case, day, df_)
            event_name_list = list(df_.loc[(df_['id_case'] == id_case) & (df_['service_name'] == service_name) & (
                        df_['day'] == day), 'event_name'])
            if ind_last_moved:
                event_list.append((event_name_list, curr_inds, id_case, ind_last_moved))
            # print(df_.loc[(df_['id_case'] == id_case) & (df_['service_name'] == service_name) & (df_['day'] == day), ['event_name', 'start_date'	,'event_start_date'	,'event_end_date']])

        df_times = pd.DataFrame([])

        if len(event_list) > 0:
            for event in event_list:
                if len(event[0]) > 1:

                    service_start = find_service_time(df_, event[1][-1])
                    service_end = find_depart_time(df_, event[1][-1])

                    if is_dispatcher(event[2], df_, event[3]):

                        arrival_time = dispatcher_start_time(event[2], df_, event[3])

                        # print(arrival_time, service_start, service_end, 'dispatecher')
                    else:
                        arrival_time = moved_start_time(event[2], df_, event[3])
                        # print(arrival_time, service_start, service_end)

                    curr_ind = df_times.shape[0]
                    df_times.loc[curr_ind, 'case_id'] = event[2]
                    df_times.loc[curr_ind, 'arrival_time'] = arrival_time
                    df_times.loc[curr_ind, 'service_start'] = service_start
                    df_times.loc[curr_ind, 'service_end'] = service_end

                else:
                    pass

            df_list.append(df_times.sort_values('arrival_time'))
            # print(event, '#########')

    from matplotlib.dates import DateFormatter
    from matplotlib.dates import HourLocator

    df_list_2 = []
    df_list_2_b = []

    for curdf in tqdm(df_list16):

        if curdf.shape[0] > 15:

            curdf = curdf.sort_values(by='arrival_time')

            curdf = update_diff(curdf)

            curdf = curdf.reset_index()

            arr_services = np.array(curdf['service_end'].sort_values())

            for ind in range(arr_services.shape[0]):
                curdf.loc[ind, 'service_end'] = arr_services[ind]

            curdf = update_diff(curdf)

            for ind in range(1, curdf.shape[0]):

                if pd.to_datetime(curdf.loc[ind, 'arrival_time']) < pd.to_datetime(curdf.loc[ind - 1, 'service_end']):
                    curdf.loc[ind, 'service_start'] = curdf.loc[ind - 1, 'service_end']

                else:
                    curdf.loc[ind, 'service_start'] = curdf.loc[ind, 'arrival_time']

            curdf = update_diff(curdf)

            if pd.to_datetime(curdf.loc[1, 'arrival_time']) > pd.to_datetime(curdf.loc[0, 'service_start']):
                curdf.loc[0, 'arrival_time'] = curdf.loc[0, 'service_start']

            else:
                curdf.loc[0, 'service_start'] = curdf.loc[0, 'arrival_time']

            df_with_cust = compute_num_cust(curdf)
            df_list_2_b.append(curdf)
            df_list_2.append(df_with_cust)

    from matplotlib.dates import DateFormatter
    from matplotlib.dates import HourLocator

    df_list_1 = []
    df_list_1_b = []

    for curdf in tqdm(df_list):

        if curdf.shape[0] > 15:

            curdf = curdf.sort_values(by='arrival_time')

            curdf = update_diff(curdf)

            curdf = curdf.reset_index()

            arr_services = np.array(curdf['service_end'].sort_values())

            for ind in range(arr_services.shape[0]):
                curdf.loc[ind, 'service_end'] = arr_services[ind]

            curdf = update_diff(curdf)

            for ind in range(1, curdf.shape[0]):

                if pd.to_datetime(curdf.loc[ind, 'arrival_time']) < pd.to_datetime(curdf.loc[ind - 1, 'service_end']):
                    curdf.loc[ind, 'service_start'] = curdf.loc[ind - 1, 'service_end']

                else:
                    curdf.loc[ind, 'service_start'] = curdf.loc[ind, 'arrival_time']

            curdf = update_diff(curdf)

            if pd.to_datetime(curdf.loc[1, 'arrival_time']) > pd.to_datetime(curdf.loc[0, 'service_start']):
                curdf.loc[0, 'arrival_time'] = curdf.loc[0, 'service_start']

            else:
                curdf.loc[0, 'service_start'] = curdf.loc[0, 'arrival_time']

            df_list_1_b.append(curdf)

            df_with_cust = compute_num_cust(curdf)

            df_list_1.append(df_with_cust)

    df_list_tot = df_list_2 + df_list_1
    df_list_tot_b = df_list_1_b + df_list_2_b

    ser_list_list = []

    for ind in tqdm(range(10000)):

        for df_with_Service in tqdm(df_list_tot_b):
            # Filter data between two dates
            df_with_Service['lower_bound'] = pd.to_datetime('09:30:00')
            without_date = df_with_Service['lower_bound'].apply(lambda d: d.time())
            df_with_Service['lower_bound'] = without_date

            df_with_Service['upper_bound'] = pd.to_datetime('11:30:00')
            without_date = df_with_Service['upper_bound'].apply(lambda d: d.time())
            df_with_Service['upper_bound'] = without_date

            df_with_Service['service_start'] = pd.to_datetime(df_with_Service['service_start'])
            df_with_Service['service_end'] = pd.to_datetime(df_with_Service['service_end'])

            df_with_Service['service_start'] = df_with_Service['service_start'].apply(pd.Timestamp)
            df_with_Service['service_end'] = df_with_Service['service_end'].apply(pd.Timestamp)

            without_date = df_with_Service['service_start'].apply(lambda d: d.time())
            df_with_Service['service_start_no_date'] = without_date
            # print(df_with_Service.shape[0])
            df_with_Service = df_with_Service.loc[
                              (df_with_Service['service_start_no_date'] > df_with_Service['lower_bound']) & (
                                          df_with_Service['service_start_no_date'] < df_with_Service['upper_bound']), :]
            # print(df_with_Service.shape[0])

            service_times = (df_with_Service['service_end'] - df_with_Service['service_start']).dt.total_seconds()

            service_times = list(service_times)
            ser_list_list.append(service_times)

        import itertools
        merged1 = list(itertools.chain(*ser_list_list))
        sertimes = np.array(merged1)

        sertimes = np.array(sertimes)

        if np.random.rand() > 0.01:
            sertimes = np.abs(sertimes + np.random.uniform(20, 100, sertimes.shape[0]))
        else:
            sertimes = np.abs(sertimes - np.random.uniform(20, 40, sertimes.shape[0]))

        from datetime import datetime, time, timedelta

        times = []
        ts = datetime(2017, 7, 17, 9, 30, 0)
        while ts <= datetime(2017, 7, 17, 11, 30, 0):
            times.append(time(ts.hour, ts.minute, ts.second))
            ts += timedelta(minutes=sertimes.mean() / 60)

        df_time_stamps = pd.DataFrame([])
        df_time_stamps['times'] = times
        df_time_stamps.iloc[5, 0]

        sermoms = []
        for mom in range(1, 6):
            sermoms.append(((sertimes / sertimes.mean()) ** mom).mean())
        sermoms_arr = np.array(sermoms)

        time_dict = {}
        for time_ in range(df_time_stamps.shape[0]):
            time_dict[time_] = np.zeros(20)

        merged1 = []
        for dff4 in tqdm(df_list_tot):
            result = [(epoch_num, give_num_at_end_epoch(dff4, time_)) for epoch_num, time_ in
                      enumerate(df_time_stamps['times'])]
            resultDictionary = dict((x, y) for x, y in result)
            merged1.append(resultDictionary)

        for resultDictionary in merged1:
            for time1 in resultDictionary.keys():
                time_dict[time1][resultDictionary[time1]] += 1

        ## Converting into probabilities

        prob_queue_arr = np.array([])
        time_dict_ = time_dict[0]
        for t in range(len(time_dict)):
            probs = (time_dict[t] / time_dict[t].sum())
            if t == 0:
                prob_queue_arr = np.array(probs).reshape(1, probs.shape[0])
            else:
                prob_queue_arr = np.concatenate((prob_queue_arr, np.array(probs).reshape(1, probs.shape[0])), axis=0)



        arrival_dict = {}
        for time_epoch in range(df_time_stamps.shape[0]-1):
            arrival_dict[time_epoch] = []

        for dff4 in tqdm(df_list_tot):

            for inter_val in range(df_time_stamps.shape[0] - 1):

                dff4['lb'] = df_time_stamps.iloc[inter_val, 0]  # pd.to_datetime(time_list[inter_val])

                dff4['ub'] = df_time_stamps.iloc[inter_val + 1, 0]  # pd.to_datetime(time_list[inter_val+1])

                inds = np.array(dff4.loc[
                                    (dff4['event_type'] == 'Arrival') & (dff4['time_stamp_without_date'] >= dff4['lb']) & (
                                                dff4['time_stamp_without_date'] < dff4['ub'])].index)

                for indcurr in inds:

                    temp_df = pd.DataFrame([])

                    for ind in range(indcurr + 1, dff4.shape[0]):
                        if dff4.loc[ind, 'event_type'] == 'Arrival':
                            temp_df.loc[0, 'second_arrival'] = dff4.loc[ind, 'time_stamp']
                            temp_df.loc[0, 'first_arrival'] = dff4.loc[indcurr, 'time_stamp']

                            temp_df['second_arrival'] = pd.to_datetime(temp_df['second_arrival'],
                                                                       format='%Y-%m-%d %H:%M:%S')
                            temp_df['first_arrival'] = pd.to_datetime(temp_df['first_arrival'], format='%Y-%m-%d %H:%M:%S')

                            inter_arrival = (temp_df['second_arrival'] - temp_df['first_arrival']).dt.total_seconds().item()
                            arrival_dict[inter_val].append(inter_arrival)
                            # print(inter_arrival, inter_val)
                            break

        for t in range(0, df_time_stamps.shape[0] - 1):
            if np.random.rand() < 0.5:
                arrival_dict[t] = np.abs(arrival_dict[t] + np.random.uniform(0, 100, len(arrival_dict[t])))
            else:
                arrival_dict[t] = np.abs(arrival_dict[t] - np.random.uniform(0, 20, len(arrival_dict[t])))

        moms_time_dict = {}
        for t in range(0, df_time_stamps.shape[0] - 1):
            curr_moms = []
            for mom in range(1, 6):
                curr_moms.append(((np.array(arrival_dict[t]) / sertimes.mean()) ** mom).mean())
            moms_time_dict[t] = np.array(curr_moms)

        res_input = np.array([])
        initial = prob_queue_arr[0, :]
        prob_queue_arr = np.array([])
        # time_dict, arrival_rates, model_inputs, initial = pkl.load(open(os.path.join(path, files_list[ind]), 'rb'))
        for t in range(0, len(time_dict) - 1):
            arr_input = np.concatenate((np.log(sermoms_arr), np.log(moms_time_dict[t]), np.array([t]), initial[:15]),
                                       axis=0)
            arr_input = arr_input.reshape(1, arr_input.shape[0])
            probs = (time_dict[t] / time_dict[t].sum())
            probs = np.concatenate((probs, np.zeros(31)), axis=0)
            if res_input.shape[0] == 0:
                res_input = arr_input
                prob_queue_arr = np.array(probs).reshape(1, probs.shape[0])
            else:
                res_input = np.concatenate((res_input, arr_input), axis=0)
                prob_queue_arr = np.concatenate((prob_queue_arr, np.array(probs).reshape(1, probs.shape[0])), axis=0)

        initial = prob_queue_arr[0, :]

        all_arrivals = 1 / np.exp(res_input[:, 5])
        num_groups = all_arrivals.shape[0]
        df_input_rates = pd.DataFrame([])

        for ind in range(all_arrivals.shape[0]):
            df_input_rates.loc[ind, 'arrival_rate'] = all_arrivals[ind]
            df_input_rates.loc[ind, 'time'] = ind
            df_input_rates.loc[ind, 'arrival_code'] = ind

        rates_dict_rate_code = {}
        rate_dict_code_rate = {}

        for ind, rate in enumerate(all_arrivals):
            rates_dict_rate_code[rate] = ind
            rate_dict_code_rate[ind] = rate

        ##### Computing service time and inter_arrival PH distributions

        a = PHFromTrace(sertimes / sertimes.mean(), 15)

        s_service, A_service = a[0], a[1]
        moms_service = np.array(compute_first_n_moments(s_service, A_service, 5)).flatten()
        s_service, A_service = a[0], a[1]
        num_samples = 100000
        services = SamplesFromPH(ml.matrix(s_service), A_service, int(num_samples))
        services.mean()

        arrival_ph_dict = {}

        for t in range(len(time_dict) - 1):
            curr_dat = np.array(arrival_dict[t] / sertimes.mean())
            a = PHFromTrace(curr_dat, 20)
            arrival_ph_dict[t] = (np.array(a[0]), np.array(a[1]))
            x_vals = np.linspace(0, 8, 200)
            # y_vals = compute_pdf_within_range(x_vals, a[0], a[1])
            # plt.figure()
            # plt.hist(curr_dat, density = True, bins = 65)
            # plt.plot(x_vals, np.array(y_vals).flatten())
            # plt.show()

        arrival_dict1 = {}

        for t in tqdm(range(len(time_dict) - 1)):
            s = arrival_ph_dict[t][0]
            A = arrival_ph_dict[t][1]
            moms_arrivals = np.array(compute_first_n_moments(s, A, 5)).flatten()
            arrivals = SamplesFromPH(ml.matrix(s), A, int(num_samples))
            arrival_dict1[t] = ((s, A, moms_arrivals, arrivals), all_arrivals[t])

        vector_lenght = 60
        num_cycles = int(vector_lenght / all_arrivals.shape[0])
        all_arrivals1 = np.tile(all_arrivals, num_cycles + 1)[:vector_lenght]
        df_input_rates = pd.DataFrame([])
        df_input_rates['arrival_rate'] = all_arrivals1
        df_input_rates['time'] = np.arange(vector_lenght)
        df_input_rates['arrival_code'] = np.tile(np.arange(all_arrivals.shape[0]), num_cycles + 1)[:vector_lenght]

        sim_input = (
        all_arrivals, num_groups, df_input_rates, rates_dict_rate_code, rate_dict_code_rate, s_service, A_service,
        moms_service, services, arrival_dict1, initial)
        now = datetime.now()
        np.random.seed(now.microsecond)

        mod_number = np.random.randint(1, 1000000000)

        pkl.dump(sim_input, open('/scratch/eliransc/sim_sets_ichilov/sim_input_Ichilov_'+str(mod_number)+'.pkl','wb'))


if __name__ == '__main__':

    main()