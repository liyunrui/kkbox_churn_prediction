#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

'''
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import utils # written by author
from glob import glob
from datetime import datetime, timedelta
import multiprocessing as mp
import gc # for automatic releasing memory

input_col = ['msno','date','num_25','num_50','num_75','num_985','num_100']



#==============================================================================
# def
#==============================================================================
def make(T):
    """
    T = 0
    folder = 'trainW-0'
    """

    if T == -1:
        folder = 'test'
        user_logs = utils.read_multiple_csv('../feature/{}/compressed_user_logs'.format(folder),input_col) 
        user_logs = pd.concat([user_logs,pd.read_csv('../input/user_logs_v2.csv', parse_dates = ['date'])[input_col]],
        ignore_index=True) 
    else:
        folder = 'trainW-'+str(T)
        user_logs = utils.read_multiple_csv('../feature/{}/compressed_user_logs'.format(folder), input_col)
    #user_logs = user_logs[user_logs.msno == 'Pz51LVoS9ENG1kNHQyrJ3gG8A163pyHi+gyvN2p+1nM=']
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(user_logs)
    gc.collect()
    print ('shape1:', user_logs.shape)

    #get_ratio
    user_logs.loc[:,"num_25":"num_100"] = user_logs.loc[:,"num_25":"num_100"].div(user_logs.loc[:,"num_25":"num_100"].sum(axis=1), axis=0)
    user_logs.rename(columns = {'num_25':'num_25_ratio','num_50':'num_50_ratio',
                           'num_75':'num_75_ratio','num_985':'num_985_ratio',
                           'num_100':'num_100_ratio'}, inplace =True)
    
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(user_logs)
    gc.collect()
    ##################################################
    # write
    ##################################################
    path = '../feature/{}/user_logs_listening_freq'.format(folder)
    gc.collect()
    utils.to_multiple_csv(user_logs, path, split_size = 10)
    print ('{0} done'.format(T))
##################################################
# Main
##################################################
s = time.time()
make(0)
make(1)
make(2)
make(-1)

# mp_pool = mp.Pool(4) # going into the pool, python will do multi processing automatically.
# mp_pool.map(make, [0,1,2,-1])

e = time.time()
print (e-s)






