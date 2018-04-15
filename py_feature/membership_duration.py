#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

it took 45 mins 
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
input_col = ['msno','transaction_date','membership_duration']
membership_loyalty = utils.read_multiple_csv('../input/preprocessed_data/transactions_date_base',input_col) # 20,000,000

#membership_loyalty = membership_loyalty.head(n = 500)

##################################################
# Convert string to datetime format
##################################################
membership_loyalty['transaction_date']  = membership_loyalty.transaction_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

#==============================================================================
print('reduce memory')
#==============================================================================
utils.reduce_memory(membership_loyalty)

def near(x, keep = 5):
    return x.tail(keep)

#==============================================================================
# def
#==============================================================================
def make(T):
	"""
	T = 0
	folder = 'trainW-0'
	"""

	if T ==-1:
	    folder = 'test'
	    train = pd.read_csv('../input/sample_submission_v2.csv') # 此train代表的是test的user
	else:
	    folder = 'trainW-'+str(T)
	    train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[['msno','is_churn']]

	# the following style is silly, but it's all for saving memory 
	if T == 0:
		df = pd.merge(train, 
	    membership_loyalty[(membership_loyalty.transaction_date < datetime.strptime('2017-03-01', '%Y-%m-%d'))], 
	    on=['msno'], 
	    how='left')
		del train
	elif T == 1:
	    # w = 1:使用2月之前的資料當作history
	    df = pd.merge(train, 
	    	membership_loyalty[(membership_loyalty.transaction_date < datetime.strptime('2017-02-01', '%Y-%m-%d'))],
	    	on=['msno'], 
	    	how='left') 
	    del train
	elif T == 2:
	    # w = 2:使用1月之前的資料當作history
	    df = pd.merge(train, 
	    	membership_loyalty[(membership_loyalty.transaction_date < datetime.strptime('2017-01-01', '%Y-%m-%d'))],
	    	on=['msno'], 
	    	how='left') 
	    del train
	elif T == -1:
	    # w = -1:使用4月之前的資料當作history
	    df = pd.merge(train, 
	    	membership_loyalty[(membership_loyalty.transaction_date < datetime.strptime('2017-04-01', '%Y-%m-%d'))],
	    	on='msno', 
	    	how='left') 
	    del train
	##################################################
	# All history
	##################################################
	#df = df.dropna()
	#core
	tbl = df.groupby('msno').membership_duration.mean().to_frame()
	tbl.columns = ['membership_duration-mean']
	tbl['membership_duration-min'] = df.groupby('msno').membership_duration.min()
	tbl['membership_duration-max'] = df.groupby('msno').membership_duration.max()
	tbl['membership_duration-median'] = df.groupby('msno').membership_duration.median()
	tbl['membership_duration-std'] = df.groupby('msno').membership_duration.std()
	tbl['transactions_count'] = df.groupby('msno').membership_duration.size()
	# is_churn: missing
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/membership_duration.csv'.format(folder), index = False)
	del tbl
	gc.collect()
	##################################################
	# near 5
	##################################################
	#core
	df_ = df.groupby('msno').apply(near,5).reset_index(drop = True)
	tbl = df_.groupby('msno').membership_duration.mean().to_frame()
	tbl.columns = ['membership_duration-mean_n5']
	tbl['membership_duration-min_n5'] = df_.groupby('msno').membership_duration.min()
	tbl['membership_duration-max_n5'] = df_.groupby('msno').membership_duration.max()
	tbl['membership_duration-median_n5'] = df_.groupby('msno').membership_duration.median()
	tbl['membership_duration-std_n5'] = df_.groupby('msno').membership_duration.std()
	tbl.reset_index(inplace = True)
	del df_
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/membership_duration_n5.csv'.format(folder), index = False)
	del tbl
	gc.collect()
	##################################################
	# only one prvious order
	##################################################
	#core
	df_ = df.groupby('msno').apply(near,1).reset_index(drop = True)
	tbl = df_.groupby('msno').membership_duration.mean().to_frame()
	tbl.columns = ['membership_duration-mean_n1']
	tbl['membership_duration-min_n1'] = df_.groupby('msno').membership_duration.min()
	tbl['membership_duration-max_n1'] = df_.groupby('msno').membership_duration.max()
	tbl['membership_duration-median_n1'] = df_.groupby('msno').membership_duration.median()
	tbl['membership_duration-std_n1'] = df_.groupby('msno').membership_duration.std()
	tbl.reset_index(inplace = True)
	del df_
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/membership_duration_n1.csv'.format(folder), index = False)
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
# mp_pool = mp.Pool(4) # going into the pool, python will do multi processing automatically.
# mp_pool.map(make, [0,1])

e = time.time()
print (e-s)




