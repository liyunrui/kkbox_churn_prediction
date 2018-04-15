#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

# 切割transaction.csv 成小塊的資料, 為了解決記憶體過大的問題
'''

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import utils # written by author
from glob import glob
from datetime import datetime

##################################################
#transactions 
##################################################

##################################################
#transactions 
##################################################


transactions = pd.concat([pd.read_csv('../input/transactions.csv'),
			   pd.read_csv('../input/transactions_v2.csv')],
			   ignore_index=True)
##################################################
# Changing the format of dates in YYYY-MM-DD in transaction data set
##################################################

transactions['transaction_date'] = transactions.transaction_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date())
transactions['membership_expire_date'] = transactions.membership_expire_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date())

##################################################
# Data Cleaning: Remove very early expiration data
##################################################

# transaction_date from 2015/01/01 to 2017/3/31
# membership_expire_date from 1970/01/01 to 2036/10/15

very_early_expiration_date = datetime.date(datetime.strptime('20190101', '%Y%m%d')) # datetime.date() : Convert datetime.datetime to datetime.date
very_late_expiration_date = datetime.date(datetime.strptime('20140101', '%Y%m%d'))
# filter
transactions = transactions[(transactions['membership_expire_date'] < very_early_expiration_date) & (very_late_expiration_date < transactions['membership_expire_date'] )]

transactions.sort_values(['msno','transaction_date'], inplace=True)
transactions.reset_index(drop=True, inplace=True)


# write
path = '../input/preprocessed_data/transactions'

utils.to_multiple_csv(transactions,path, split_size = 4)

