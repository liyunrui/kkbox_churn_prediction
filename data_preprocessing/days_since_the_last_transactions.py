#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

it took 7.43 hours
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
# Load  
##################################################
col = ['msno','plan_list_price','payment_plan_days','actual_amount_paid','payment_method_id','transaction_date',]
transactions = utils.read_multiple_csv('../input/preprocessed_data/transactions', col) # 20,000,000

#transactions = transactions.head(n = 5000)

input_col = ['msno','transaction_date','t-1_transaction_date','t-1_membership_expire_date','t-1_actual_amount_paid',
             'order_number_rev','t-1_payment_method_id','t-1_plan_list_price','t-1_payment_plan_days']
tran_time_diff = utils.read_multiple_csv('../input/preprocessed_data/transactions_time_diff', input_col)
tran_time_diff = tran_time_diff.dropna() # drop columns whose oder_number == 1

#tran_time_diff = tran_time_diff.head(n = 5000)

#==============================================================================
print('reduce memory')
#==============================================================================
utils.reduce_memory(transactions)
utils.reduce_memory(tran_time_diff)

##################################################
# Convert string to datetime format
##################################################
transactions['transaction_date']  = transactions.transaction_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
tran_time_diff['transaction_date']  = tran_time_diff['transaction_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
tran_time_diff['t-1_membership_expire_date']  = tran_time_diff['t-1_membership_expire_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
tran_time_diff['t-1_transaction_date']  = tran_time_diff['t-1_transaction_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

gc.collect()

print ('After Loading ...')

# create features
def days_since_the_last_expiration(x):
    # x: each row of dataframe, series
    x['days_since_the_last_expiration'] = [i.days for i in (x.transaction_date - x['t-1_membership_expire_date'])]
    return x.drop(['t-1_membership_expire_date'], axis=1)
def days_since_the_last_subscription(x):
    x['days_since_the_last_subscription'] = [i.days for i in (x.transaction_date - x['t-1_transaction_date'])]
    return x.drop(['transaction_date','t-1_transaction_date'], axis=1)
def is_subscribe_early(x):
    x['is_subscribe_early'] = [1 if i <0 else 0 for i in x.days_since_the_last_expiration]
    return x
def creat_loyalty_trend(x):
    drop_col = ['payment_method_id','plan_list_price','payment_plan_days','actual_amount_paid',
                't-1_plan_list_price','t-1_payment_plan_days','t-1_payment_method_id','t-1_actual_amount_paid'] # 這些col是計算完之後不需要output的都刪掉
    # date
    x['days_since_the_last_expiration-cumsum'] = x.days_since_the_last_expiration.cumsum()
    x['days_since_the_last_expiration_ratio'] = x.days_since_the_last_expiration.cumsum()/ [ i+1 for i,j in enumerate(list(reversed(x.order_number_rev.tolist())))]
    x['days_since_the_last_subscription_ratio'] = x.days_since_the_last_subscription.cumsum()/ [ i+1 for i,j in enumerate(list(reversed(x.order_number_rev.tolist())))]
    #x['days_since_the_last_expiration_diff'] = x.days_since_the_last_expiration - x.days_since_the_last_expiration.shift(1)
    x['days_since_the_first_subscription'] = x.days_since_the_last_subscription.cumsum()
    # payment_method
    x['do_change_payment_method'] = [1 if (p_m - t_1_p_m) != 0 else 0 for p_m, t_1_p_m in x[['payment_method_id','t-1_payment_method_id']].values]
    # plan_list_price(這次訂單,有選擇更高的價錢的方案麼)
    x['do_spend_more_money'] = [p_price - t_1_p_price for p_price, t_1_p_price in x[['plan_list_price','t-1_plan_list_price']].values]
    # payment_plan_days(這次訂單,有選擇天數更高的方案麼)
    x['do_extend_payment_days'] = [p_p_days - t_1_p_days for p_p_days, t_1_p_days in x[['payment_plan_days','t-1_payment_plan_days']].values]
    # (這次訂單,真實付的錢有比上次多麽)
    x['do_paid_more'] = [a_paid - a_paid for a_paid, a_paid in x[['actual_amount_paid','t-1_actual_amount_paid']].values]
    return x.drop(drop_col, axis=1)

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
        test = pd.read_csv('../input/sample_submission_v2.csv')
    else:
        col = ['msno','w']
        folder = 'trainW-'+str(T)
        train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[col]
    #merge data

    if T == 0:
        # w = 0:使用3月之前的資料當作history
        df = pd.merge(train, 
        transactions[(transactions.transaction_date < datetime.strptime('2017-03-01', '%Y-%m-%d'))], 
        on='msno', 
        how='left') # 此時msno就是不重複
        del train
    elif T == 1:
        # w = 1:使用2月之前的資料當作history
        df = pd.merge(train, 
        transactions[(transactions.transaction_date < datetime.strptime('2017-02-01', '%Y-%m-%d'))],
        on='msno', 
        how='left') # 此時msno就是不重複
        del train
    elif T == 2:
        # w = 2:使用1月之前的資料當作history
        df = pd.merge(train, 
        transactions[(transactions.transaction_date < datetime.strptime('2017-01-01', '%Y-%m-%d'))],
        on='msno', 
        how='left') # 此時msno就是不重複
        del train
    elif T == -1:
        # w = -1:使用4月之前的資料當作history
        df = pd.merge(test, 
        transactions[(transactions.transaction_date < datetime.strptime('2017-04-01', '%Y-%m-%d'))],
        on='msno', 
        how='left') # 此時msno就是不重複
        del test
        df['w'] = T 
    gc.collect()

    df = pd.merge(df, tran_time_diff, on =['msno','transaction_date'], how='left')  
    
    #df = df.dropna().reset_index(drop =True)

    # creating features
    df = df.groupby('msno').apply(days_since_the_last_expiration)
    df = df.groupby('msno').apply(days_since_the_last_subscription)
    df = df.groupby('msno').apply(is_subscribe_early) 
    df = df.groupby('msno').apply(creat_loyalty_trend)

    output_col = ['msno','w','days_since_the_last_expiration',
            'days_since_the_last_subscription','is_subscribe_early','order_number_rev',
            'days_since_the_last_expiration-cumsum',
            'days_since_the_last_expiration_ratio',
            'days_since_the_last_subscription_ratio',
            'days_since_the_first_subscription',
                  'do_change_payment_method',
                  'do_spend_more_money',
                  'do_extend_payment_days',
                  'do_paid_more',
            ]
    df = df.dropna().reset_index(drop =True) # for the following reduce memory
    df = df[output_col] # msno is not unique
    gc.collect()
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(df)
 
    #write
    path = '../feature/{}/days_since_the_last_transactions'.format(folder)
    utils.to_multiple_csv(df, path, split_size = 4)
    print ('{0} done'.format(T))  
    del df
    gc.collect()

# 優化措施(Speed)    
# 多花一些記憶題把它寫下來(只會用一次), 如果可以避免之後反覆I/O,這樣反而是省時間的,
##################################################
# Main
##################################################
s = time.time()

# make(0)
# make(1)
# make(2)
make(-1)

# mp_pool = mp.Pool(4) # going into the pool, python will do multi processing automatically.
# mp_pool.map(make, [0,1]) 
# mp_pool.map(make, [2,-1]) 

gc.collect()

e = time.time()
print (e-s)







