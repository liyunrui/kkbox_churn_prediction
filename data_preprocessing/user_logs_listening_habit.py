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

input_col = ['msno','date','total_secs']

# def habit_discrimination(x):
#     # 0.5: 一天聽12小時
#     # 0.333: 一天聽8小時
#     # 0.125: 一天聽3小時
#     # 0.0416: 一天聽1小時
#     if x > 0.5:
#         return 1 # 重度kkbox依賴者
#     elif 0.5 > x > 0.125:
#         return 2
#     elif 0.125 > x > 0.0416:
#         return 3
#     else:
#         return 4
def completed_vs_incompleted_songs(x):
    x['num_completed_songs'] = x.num_100 + x.num_985
    x['num_incompleted_songs'] = x.num_25 + x.num_50 + x.num_75
    return x
def make_order_number(x):
    x['order_number'] = [i+1 for i in range(x.shape[0])]
    return x

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
        ignore_index = True) 
        # user_logs_v2.csv: only inclue data of March, it's for testing set.
        #user_logs.sort_values(by = ['msno', 'date'], inplace = True)
        #這邊記憶體會激升, 速度會變慢因為concat and sort_values,現在問題是有需要sort_values麼?有groupby就不需要
    else:
        folder = 'trainW-'+ str(T)
        user_logs = utils.read_multiple_csv('../feature/{}/compressed_user_logs'.format(folder), input_col)
    
    #user_logs = user_logs[user_logs.msno == 'Pz51LVoS9ENG1kNHQyrJ3gG8A163pyHi+gyvN2p+1nM=']
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(user_logs)
    gc.collect()
    print ('shape1:', user_logs.shape)
    # core
    user_logs['total_secs_percentage'] = user_logs.total_secs.apply(lambda x: x / (24*60*60))
    #user_logs['listening_habit_zone'] = user_logs.total_secs_percentage.apply(habit_discrimination)

    user_logs['num_of_time_the_user_has_logged_in'] = user_logs.groupby('msno').total_secs.cumsum() # make this line faster
    user_logs.drop('total_secs', axis = 1, inplace = True)
    user_logs = user_logs.groupby('msno').apply(make_order_number) # make this line faster
    user_logs['num_of_time_the_user_has_logged_in_ratio'] = user_logs.num_of_time_the_user_has_logged_in / user_logs.order_number
    user_logs.drop('order_number', axis = 1, inplace = True)
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(user_logs)
    print ('shape2:', user_logs.shape)
    ##################################################
    # write
    ##################################################
    path = '../feature/{}/user_logs_listening_habit'.format(folder)
    gc.collect()
    utils.to_multiple_csv(user_logs,path, split_size = 8)
    del user_logs
    gc.collect()
    print ('{0} done'.format(T))
##################################################
# Main
##################################################
s = time.time()
#make(0)
make(1)
make(2)
make(-1)

# mp_pool = mp.Pool(4) # going into the pool, python will do multi processing automatically.
# mp_pool.map(make, [0,1,2,-1])

e = time.time()
print (e-s)


















