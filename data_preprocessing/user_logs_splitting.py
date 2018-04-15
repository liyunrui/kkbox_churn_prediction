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


##################################################
# Load user_logs
##################################################
# chunksize=10,000,000 (10 millions)
user_logs_iter = pd.read_csv("../input/user_logs.csv", chunksize=1000, 
            iterator=True, low_memory=False, parse_dates=['date'])

#==============================================================================
#def
#==============================================================================

def make(T):
    if T == -1:
        folder = 'test'
        test = pd.read_csv('../input/sample_submission_v2.csv')
        #user_logs_march = pd.read_csv('../input/user_logs_v2.csv')
        #==============================================================================
        print('reduce memory')
        #==============================================================================
        utils.reduce_memory(test)
        #utils.reduce_memory(user_logs_march)
        
        valid_msno = test['msno'].as_matrix()
    else:
        col = ['msno']
        folder = 'trainW-'+str(T)
        train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[col]     
        #==============================================================================
        print('reduce memory')
        #==============================================================================
        utils.reduce_memory(train)
        
        valid_msno = train['msno'].as_matrix()  
    ##################################################
    #core
    ##################################################
    c = 0
    chunked_user_logs = pd.DataFrame() 
    start_time = time.time()

    for df in user_logs_iter:
        c += 1
        #==============================================================================
        print('reduce memory')
        #==============================================================================
        utils.reduce_memory(df)

        df = df[df['msno'].isin(valid_msno)] #找出chunked_user_logs中msno有在valid_msno的dataframe      
        df.sort_values(by = ['msno','date']).reset_index(drop = True)
        gc.collect()
        ##################################################
        # Convert string to datetime format
        ##################################################
        # 這邊每一個loop花5分鐘的話
        #print ('befor',df)
        df['date']  = df.date.apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
        #print ('after',df)
        ##################################################
        # Filtering accroding to w
        ##################################################
        if T == 0:
            # w = 0:使用3月之前的資料當作history
            df = df[ (df.date < datetime.strptime('2017-03-01', '%Y-%m-%d'))]
        elif T == 1:
            # w = 1:使用2月之前的資料當作history
            df = df[ (df.date < datetime.strptime('2017-02-01', '%Y-%m-%d'))]
        elif T == 2:
            # w = 2:使用1月之前的資料當作history
            df = df[ (df.date < datetime.strptime('2017-01-01', '%Y-%m-%d'))]
        elif T == -1:
            # w = -1:使用4月之前的資料當作history
            df = df[ (df.date < datetime.strptime('2017-04-01', '%Y-%m-%d'))]
        chunked_user_logs = chunked_user_logs.append(df)
        print("Loop ",c,"took %s seconds" % (time.time() - start_time))
        #print("User Logs {} df: {}".format(str(c), df.shape))
        # if c > 5:
        #     break
    print("Shape of Chunked User Logs: {}".format(chunked_user_logs.shape))
    ##################################################
    # write
    ##################################################
    path = '../feature/{}/compressed_user_logs'.format(folder)
    gc.collect()
    utils.to_multiple_csv(chunked_user_logs,path, split_size = 40)

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
