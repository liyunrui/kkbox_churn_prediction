#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

It took 5.6666 hours

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
# Load transaction 
##################################################
#input_col = []
transactions = utils.read_multiple_csv('../input/preprocessed_data/transactions')
#print ('shape of transactions',transactions.shape)
#transactions = transactions.head(n = 500)

#==============================================================================
print('reduce memory')
#==============================================================================
utils.reduce_memory(transactions)
##################################################
# Convert string to datetime format
##################################################
transactions['membership_expire_date']  = transactions.membership_expire_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
transactions['transaction_date']  = transactions.transaction_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))



print ('After Loading ...')
##################################################
# 善用time(時間差)來區隔交易日期時間點的行為
##################################################
def make_order_number(x):
    # for near 5
    # 1:代表最靠近現在,數字越大代表是越過去的歷史紀錄
    x['order_number_rev'] = list(reversed([i+1 for i in range(x.shape[0])]))
    return x
def t_diff(x, time_diff):
    # x : the dataframe belong to a each specific user
    # time_diff: 2,3,4,...
    output_col = ['msno','transaction_date','t-1_transaction_date','t-1_membership_expire_date',
              't-1_payment_method_id','t-1_payment_plan_days',
             't-1_plan_list_price','t-1_actual_amount_paid',
             't-1_is_auto_renew','t-1_is_cancel'
             ]
    for i in range(1, time_diff):
        x['t-{}_transaction_date'.format(i)] = x['transaction_date'].shift(i)
        x['t-{}_membership_expire_date'.format(i)] = x['membership_expire_date'].shift(i)
        x['t-{}_payment_method_id'.format(i)] = x['payment_method_id'].shift(i)
        x['t-{}_payment_plan_days'.format(i)] = x['payment_plan_days'].shift(i)
        x['t-{}_plan_list_price'.format(i)] = x['plan_list_price'].shift(i)
        x['t-{}_actual_amount_paid'.format(i)] = x['actual_amount_paid'].shift(i)
        x['t-{}_is_auto_renew'.format(i)] = x['is_auto_renew'].shift(i)
        x['t-{}_is_cancel'.format(i)] = x['is_cancel'].shift(i)
        #生成想要的output_col之後,把用不到的input_col給drop掉啊。
    return x[output_col]


##################################################
# Main
##################################################
s = time.time()

transactions = transactions.groupby('msno').apply(t_diff, 2)
transactions = transactions.groupby('msno').apply(make_order_number)

gc.collect()
# write
path = '../input/preprocessed_data/transactions_time_diff'

output_col = ['msno','transaction_date','t-1_transaction_date','t-1_membership_expire_date',
              't-1_payment_method_id','t-1_payment_plan_days',
             't-1_plan_list_price','t-1_actual_amount_paid',
             't-1_is_auto_renew','t-1_is_cancel',
             'order_number_rev'
             ]


transactions = transactions[output_col] 
#==============================================================================
print('reduce memory')
#==============================================================================
transactions = transactions.dropna() # 此動作會把每個user的第一筆drop掉, 但因為第一筆訂單本來就沒有時間差的feature, 所以沒有少資訊。
utils.reduce_memory(transactions)

gc.collect()
utils.to_multiple_csv(transactions,path, split_size = 4)

e = time.time()
print (e-s)




