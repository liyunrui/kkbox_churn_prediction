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

input_col = ['msno','date','num_25','num_50','num_75','num_985','num_100','num_unq']



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
        #user_logs.sort_values(by = ['msno', 'date'],inplace = True)
    else:
        folder = 'trainW-'+str(T)
        user_logs = utils.read_multiple_csv('../feature/{}/compressed_user_logs'.format(folder), input_col)
    #user_logs = user_logs[user_logs.msno == 'Pz51LVoS9ENG1kNHQyrJ3gG8A163pyHi+gyvN2p+1nM=']
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(user_logs)
    print ('shape1:', user_logs.shape)
    gc.collect()
    #incompleted vs completed 
    user_logs['num_completed_songs'] = user_logs.num_100 + user_logs.num_985
    user_logs['num_incompleted_songs'] = user_logs.num_25 + user_logs.num_50 + user_logs.num_75
    user_logs['completed_songs_ratio'] = user_logs.num_completed_songs/ (user_logs.num_incompleted_songs + user_logs.num_completed_songs)
    user_logs['is_satisfied'] = user_logs.completed_songs_ratio.apply(lambda x: 1 if x > 0.5 else 0)
    #num_repeated_songs
    user_logs['num_repeated_songs'] = (user_logs.num_100 + user_logs.num_985 + user_logs.num_75) / user_logs.num_unq
    user_logs.drop(['num_25','num_50','num_75','num_985','num_100','num_unq'], axis = 1, inplace = True)
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(user_logs)
    print ('shape2:', user_logs.shape)
    gc.collect()
    ##################################################
    # write
    ##################################################
    path = '../feature/{}/user_logs_listening_behavior'.format(folder)
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


















