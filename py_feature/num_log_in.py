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

def listening_longevity(x):
    # input_col = ['msno','date']
    x['listening_longevity'] = (x.iloc[-1].date - x.iloc[0].date).days
    return x

def within_n_days(x, T, n = 7):
    # n = 7, 14, 30, 60, 90,
    ##################################################
    # Filtering accroding to w
    ##################################################
    if T == 0:
        # w = 0:使用3月之前的資料當作history
        prediction_deadline = datetime.strptime('2017-03-01', '%Y-%m-%d')
    elif T == 1:
        # w = 1:使用2月之前的資料當作history
        prediction_deadline = datetime.strptime('2017-02-01', '%Y-%m-%d')
    elif T == 2:
        # w = 2:使用1月之前的資料當作history
        prediction_deadline = datetime.strptime('2017-01-01', '%Y-%m-%d')
    elif T == -1:
        # w = -1:使用4月之前的資料當作history
        prediction_deadline = datetime.strptime('2017-04-01', '%Y-%m-%d')
    # n = 7, 14, 21, 30
    check_points = prediction_deadline - timedelta(n) # 往prediction_deadline前n天內
    x = x[ (x.date >= check_points )]
    # 如果有些使用者在預測deadline前n天內沒有資料,到時候再concat, merge的時候會是nan就把它補-1
    return x

#==============================================================================
# def
#==============================================================================
def make(T):
    """
    T = 0
    folder = 'trainW-0'
    """
    input_col = ['msno','date']
    #output_col = ['msno','num_log_in','listening_longevity','log_in_ratio']

    if T == -1:
        folder = 'test'
        #label
        train = pd.read_csv('../input/sample_submission_v2.csv')[['msno']] # 此train代表的是test的user
        user_logs = utils.read_multiple_csv('../feature/{}/compressed_user_logs'.format(folder),
            input_col,
            parse_dates = ['date']) 
        user_logs = pd.concat([user_logs,pd.read_csv('../input/user_logs_v2.csv',parse_dates = ['date'])[input_col]],
        ignore_index = True) 
        #user_logs.sort_values(by = ['msno', 'date'],inplace = True)
    else:
        folder = 'trainW-'+ str(T)
        #label
        train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[['msno']] 
        user_logs = utils.read_multiple_csv('../feature/{}/compressed_user_logs'.format(folder), 
            input_col,
            parse_dates = ['date'])

    ##################################################
    # basic procedure
    ##################################################
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(user_logs)

    df = pd.merge(train,user_logs, on = 'msno', how = 'left')
    del user_logs
    gc.collect()
    print ('shape of df:', df.shape)

    ##################################################
    # All history
    ##################################################
    # count
    tbl = df.groupby('msno').date.size().to_frame()
    tbl.columns = ['num_log_in']
    tbl.reset_index(inplace = True)
    # for computing log_in_ratio
    user_logs_copy = df.groupby('msno').apply(listening_longevity)
    user_logs_copy.drop_duplicates('msno',inplace =True)

    tbl = pd.merge(tbl,user_logs_copy, on = 'msno', how = 'left')

    del user_logs_copy
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(tbl)    
    gc.collect()   
    #log_in_ratio
    tbl['log_in_ratio'] = tbl.num_log_in/tbl.listening_longevity
    tbl.drop(['date','listening_longevity'], axis = 1,inplace = True)
    # write
    tbl.to_csv('../feature/{}/num_log_in.csv'.format(folder), index = False)

    del tbl
    gc.collect()
    ##################################################
    # n = 7
    ##################################################
    df_ = df.groupby('msno').apply(within_n_days,T, n = 7).reset_index(drop = True)   
    tbl = df_.groupby('msno').date.size().to_frame()
    tbl.columns = ['num_log_in_during_t_7']
    tbl.reset_index(inplace = True)
    # for computing log_in_ratio
    user_logs_copy = df_.groupby('msno').apply(listening_longevity)
    user_logs_copy.drop_duplicates('msno',inplace =True)

    tbl = pd.merge(tbl,user_logs_copy, on = 'msno', how = 'left')

    del user_logs_copy
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(tbl)   
    gc.collect()   
    #log_in_ratio
    tbl['log_in_ratio_during_t_7'] = tbl.num_log_in_during_t_7/tbl.listening_longevity
    tbl.drop(['date','listening_longevity'], axis = 1, inplace = True)
    # write
    tbl.to_csv('../feature/{}/num_log_in_during_t_7.csv'.format(folder), index = False)

    del tbl
    gc.collect()

    ##################################################
    # n = 14
    ##################################################
    df_ = df.groupby('msno').apply(within_n_days,T, n = 14).reset_index(drop = True)   
    tbl = df_.groupby('msno').date.size().to_frame()
    tbl.columns = ['num_log_in_during_t_14']
    tbl.reset_index(inplace = True)
    # for computing log_in_ratio
    user_logs_copy = df_.groupby('msno').apply(listening_longevity)
    user_logs_copy.drop_duplicates('msno',inplace =True)

    tbl = pd.merge(tbl,user_logs_copy, on = 'msno', how = 'left')

    del user_logs_copy
    gc.collect()   
    #log_in_ratio
    tbl['log_in_ratio_during_t_14'] = tbl.num_log_in_during_t_14/tbl.listening_longevity
    tbl.drop(['date','listening_longevity'], axis = 1, inplace = True)
    # write
    tbl.to_csv('../feature/{}/num_log_in_during_t_14.csv'.format(folder), index = False)

    del tbl
    gc.collect()


    ##################################################
    # n = 30
    ##################################################
    df_ = df.groupby('msno').apply(within_n_days,T, n = 30).reset_index(drop = True)   
    tbl = df_.groupby('msno').date.size().to_frame()
    tbl.columns = ['num_log_in_during_t_30']
    tbl.reset_index(inplace = True)
    # for computing log_in_ratio
    user_logs_copy = df_.groupby('msno').apply(listening_longevity)
    user_logs_copy.drop_duplicates('msno',inplace =True)

    tbl = pd.merge(tbl,user_logs_copy, on = 'msno', how = 'left')

    del user_logs_copy
    gc.collect()   
    #log_in_ratio
    tbl['log_in_ratio_during_t_30'] = tbl.num_log_in_during_t_30/tbl.listening_longevity
    tbl.drop(['date','listening_longevity'], axis = 1, inplace = True)
    # write
    tbl.to_csv('../feature/{}/num_log_in_during_t_30.csv'.format(folder), index = False)

    del tbl
    gc.collect()
    ##################################################
    # n = 60
    ##################################################
    df_ = df.groupby('msno').apply(within_n_days,T, n = 60).reset_index(drop = True)   
    tbl = df_.groupby('msno').date.size().to_frame()
    tbl.columns = ['num_log_in_during_t_60']
    tbl.reset_index(inplace = True)
    # for computing log_in_ratio
    user_logs_copy = df_.groupby('msno').apply(listening_longevity)
    user_logs_copy.drop_duplicates('msno',inplace =True)

    tbl = pd.merge(tbl,user_logs_copy, on = 'msno', how = 'left')

    del user_logs_copy
    gc.collect()   
    #log_in_ratio
    tbl['log_in_ratio_during_t_60'] = tbl.num_log_in_during_t_60/tbl.listening_longevity
    tbl.drop(['date','listening_longevity'], axis = 1, inplace = True)
    # write
    tbl.to_csv('../feature/{}/num_log_in_during_t_60.csv'.format(folder), index = False)

    del tbl
    gc.collect()

    ##################################################
    # n = 90
    ##################################################
    df_ = df.groupby('msno').apply(within_n_days,T, n = 90).reset_index(drop = True)   
    tbl = df_.groupby('msno').date.size().to_frame()
    tbl.columns = ['num_log_in_during_t_90']
    tbl.reset_index(inplace = True)
    # for computing log_in_ratio
    user_logs_copy = df_.groupby('msno').apply(listening_longevity)
    user_logs_copy.drop_duplicates('msno',inplace =True)

    tbl = pd.merge(tbl,user_logs_copy, on = 'msno', how = 'left')

    del user_logs_copy
    gc.collect()   
    #log_in_ratio
    tbl['log_in_ratio_during_t_90'] = tbl.num_log_in_during_t_90/tbl.listening_longevity
    tbl.drop(['date','listening_longevity'], axis = 1, inplace = True)
    # write
    tbl.to_csv('../feature/{}/num_log_in_during_t_90.csv'.format(folder), index = False)

    del tbl
    gc.collect()



##################################################
# Main
##################################################
s = time.time()
make(0)
make(1)
make(2)
make(-1)

e = time.time()
print (e-s)


