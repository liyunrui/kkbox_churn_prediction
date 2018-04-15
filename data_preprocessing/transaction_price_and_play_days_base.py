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
input_col = ['msno','plan_list_price','actual_amount_paid','payment_plan_days']
transactions = utils.read_multiple_csv('../input/preprocessed_data/transactions',input_col)

#==============================================================================
print('reduce memory')
#==============================================================================
utils.reduce_memory(transactions)

##################################################
# discount
##################################################
transactions['discount'] = transactions['plan_list_price'] - transactions['actual_amount_paid']
transactions['is_discount'] = transactions.discount.apply(lambda x: 1 if x > 0 else 0)

##################################################
# cp value
##################################################
transactions['amt_per_day'] = transactions['actual_amount_paid'] / transactions['payment_plan_days']
transactions['cp_value'] = transactions['plan_list_price'] / transactions['payment_plan_days']

# write
path = '../input/preprocessed_data/transaction_price_and_play_days_base'

output_col = ['msno','discount','is_discount',
              'amt_per_day','cp_value',]

transactions = transactions[output_col] 
transactions = transactions.dropna()

#==============================================================================
print('reduce memory')
#==============================================================================
utils.reduce_memory(transactions)
gc.collect()

utils.to_multiple_csv(transactions,path, split_size = 4)
