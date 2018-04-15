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
# Load transaction 
##################################################
input_col = ['msno','transaction_date','payment_plan_days', 'membership_expire_date']
transactions = utils.read_multiple_csv('../input/preprocessed_data/transactions',input_col)
#transactions = transactions.head(n = 200)
#==============================================================================
print('reduce memory')
#==============================================================================
utils.reduce_memory(transactions)

##################################################
# Convert string to datetime format
##################################################
transactions['membership_expire_date']  = transactions.membership_expire_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
transactions['transaction_date']  = transactions.transaction_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

##################################################
# For membership_loyalty
##################################################
transactions['membership_duration'] = [i.days for i in (transactions.membership_expire_date - transactions.transaction_date)]
transactions['is_membership_duration_equal_to_plan_days'] = [1 if m_d==p_d else 0 for m_d, p_d in transactions[['membership_duration','payment_plan_days']].values]
transactions['is_membership_duration_longer_than_plan_days'] = [1 if m_d > p_d else 0 for m_d, p_d in transactions[['membership_duration','payment_plan_days']].values]
transactions['days_longer_than_plan_days'] = [i if i > 0 else 0 for i in (transactions.membership_duration - transactions.payment_plan_days)]

##################################################
# 到期日在交易日之前...
##################################################

transactions['is_early_expiration'] = [1 if i.days < 0 else 0 for i in (transactions.membership_expire_date - transactions.transaction_date)]
transactions['early_expiration_days'] = [-i.days if i.days < 0 else 0 for i in (transactions.membership_expire_date - transactions.transaction_date) ]
transactions['early_expiration_days'] = transactions.early_expiration_days.apply(lambda x : x* -1 if x < 0 else x)
#==============================================================================
print('reduce memory')
#==============================================================================
utils.reduce_memory(transactions)
gc.collect()

# write
path = '../input/preprocessed_data/transactions_date_base'

output_col = ['msno','transaction_date','membership_duration','is_membership_duration_equal_to_plan_days',
              'is_membership_duration_longer_than_plan_days','days_longer_than_plan_days',
             'is_early_expiration','early_expiration_days'
             ]

transactions = transactions[output_col] 
gc.collect()
utils.to_multiple_csv(transactions,path, split_size = 4)




