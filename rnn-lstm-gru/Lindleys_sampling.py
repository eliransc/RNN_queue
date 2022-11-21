import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
import time

def compute_arrival_time(x, df):
    return df.loc[:x, 'inter_arrival'].sum()

def compute_NIS(x, df3):
    return df3.loc[:x].loc[df3.loc[:x,'Type'] == 'arrival'].shape[0]-df3.loc[:x].loc[df3.loc[:x,'Type'] == 'departure'].shape[0]

def find_num_cust_time_stamp(df, time):
    if df.loc[df['Time_stamp'] < time, :].shape[0] == 0:
        return 0
    else:
        LB = df.loc[df['Time_stamp'] < time, :].index[-1]
        return df.loc[LB, 'num_cust']



def single_sim():

    arrivals = np.random.exponential(1, 60)
    services = np.random.exponential(1.05, 60)

    waiting = [0]
    for ind in range(1, arrivals.shape[0]):
        waiting.append(max(waiting[-1 ] +services[ind-1] -arrivals[ind] ,0))
    df = pd.DataFrame([], columns = ['inter_arrival', 'service', 'waiting'])

    df['inter_arrival'] = arrivals
    df['service'] = services
    df['waiting'] = waiting
    df['index'] = df.index

    df['arrival_time'] = df['index'].apply(lambda x: compute_arrival_time(x, df ))

    df['departure_time'] = df['arrival_time' ]+ df['waiting' ] +df['service']

    df1 = pd.DataFrame([], columns = ['Time_stamp', 'Type' , 'Customer_id'])
    df2 = pd.DataFrame([], columns = ['Time_stamp', 'Type', 'Customer_id'])

    df1['Time_stamp'] = df['arrival_time']
    df1['Type'] = 'arrival'
    df1['Customer_id'] = df['index']

    df2['Time_stamp'] = df['departure_time']
    df2['Type'] = 'departure'
    df2['Customer_id'] = df['index']

    df3 = pd.concat([df1, df2])


    df3 = df3.sort_values(by=['Time_stamp'])
    df3 = df3.reset_index()
    df3['index1'] = df3.index

    df3['num_cust'] = df3['index1'].apply(lambda x: compute_NIS(x, df3))

    now = time.time()
    df3.loc[0, 'num_cust'] = 1
    for ind in range(1 ,df3.shape[0]):
        if df3.loc[ind, 'Type'] == 'arrival':
            df3.loc[ind, 'num_cust'] = df3.loc[ind -1, 'num_cust'] + 1
        else:
            df3.loc[ind, 'num_cust'] = df3.loc[ind -1, 'num_cust'] - 1
    # print('num cust took: ', time.time( ) -now)

    now = time.time()
    result = [(time_epoch, find_num_cust_time_stamp(df3, time_epoch)) for time_epoch in range(50)]
    resultDictionary = dict((x, y) for x, y in result)

    # print('dict took: ', time.time( ) -now)

    return resultDictionary


if __name__ == '__main__':

    list_of_dicts = [single_sim() for ind in tqdm(range(10000))]