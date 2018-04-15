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
# Load  
##################################################
input_col = ['msno','transaction_date','is_cancel']
transactions = utils.read_multiple_csv('../input/preprocessed_data/transactions', input_col) # 20,000,000

#transactions = transactions.head(n = 5000)
##################################################
# Convert string to datetime format
##################################################
transactions['transaction_date']  = transactions.transaction_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

#==============================================================================
print('reduce memory')
#==============================================================================

utils.reduce_memory(transactions)

def near(x, keep = 5):
    return x.tail(keep)
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

	if T ==-1:
	    folder = 'test'
	    train = pd.read_csv('../input/sample_submission_v2.csv') # 此train代表的是test的user
	else:
	    folder = 'trainW-'+str(T)
	    train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[['msno']]

	# the following style is silly, but it's all for saving memory 
	if T == 0:
		df = pd.merge(train, 
	    transactions[(transactions.transaction_date < datetime.strptime('2017-03-01', '%Y-%m-%d'))], 
	    on=['msno'], 
	    how='left')
		del train
	elif T == 1:
	    # w = 1:使用2月之前的資料當作history
	    df = pd.merge(train, 
	    	transactions[(transactions.transaction_date < datetime.strptime('2017-02-01', '%Y-%m-%d'))],
	    	on=['msno'], 
	    	how='left') 
	    del train
	elif T == 2:
	    # w = 2:使用1月之前的資料當作history
	    df = pd.merge(train, 
	    	transactions[(transactions.transaction_date < datetime.strptime('2017-01-01', '%Y-%m-%d'))],
	    	on=['msno'], 
	    	how='left') 
	    del train
	elif T == -1:
	    # w = -1:使用4月之前的資料當作history
	    df = pd.merge(train, 
	    	transactions[(transactions.transaction_date < datetime.strptime('2017-04-01', '%Y-%m-%d'))],
	    	on='msno', 
	    	how='left') 
	    del train
	##################################################
	# All history
	##################################################
	#df = df.dropna()

	df_ = df.groupby('msno').apply(make_order_number)
	#count
	cnt = df_.groupby(['msno', 'is_cancel']).size()
	cnt.name = 'cnt'
	cnt = cnt.reset_index()
	# chance
	user_onb_max = df_.groupby('msno').order_number.max().reset_index()
	user_onb_max.columns = ['msno', 'onb_max']
	user_is_cancel_min = df_.groupby(['msno', 'is_cancel']).order_number.min().reset_index()
	user_is_cancel_min.columns = ['msno', 'is_cancel', 'onb_min']

	chance = pd.merge(user_is_cancel_min, user_onb_max, on='msno', how='left')
	chance['is_cancel_chance'] = chance.onb_max - chance.onb_min +1

	tbl = pd.merge(cnt, chance, on= ['msno', 'is_cancel'], how='left')
	tbl['cancel_ratio_by_chance'] = tbl.cnt / tbl.is_cancel_chance
	# total_count
	tbl_ = df_.groupby('msno').is_cancel.sum().to_frame()
	tbl_.columns = ['cancel_total_count']
	tbl_['cancel_total_count_ratio'] = df_.groupby('msno').is_cancel.mean()
	tbl_.reset_index(inplace = True)
	tbl = pd.merge(tbl, tbl_, on = 'msno')
	col = ['msno', 'is_cancel_chance', 'cancel_ratio_by_chance','cancel_total_count','cancel_total_count_ratio']
	tbl = tbl[col]
	tbl.drop_duplicates('msno',keep = 'last',inplace = True) # 只要is_cancel == 1
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/is_cancel.csv'.format(folder), index = False)

	del tbl
	del tbl_
	del df_
	gc.collect()
	##################################################
	# near 5
	##################################################
	df_ = df.groupby('msno').apply(near,5).reset_index(drop = True)
	df_ = df_.groupby('msno').apply(make_order_number)
	#count
	cnt = df_.groupby(['msno', 'is_cancel']).size()
	cnt.name = 'cnt'
	cnt = cnt.reset_index()
	# chance
	user_onb_max = df_.groupby('msno').order_number.max().reset_index()
	user_onb_max.columns = ['msno', 'onb_max']
	user_is_cancel_min = df_.groupby(['msno', 'is_cancel']).order_number.min().reset_index()
	user_is_cancel_min.columns = ['msno', 'is_cancel', 'onb_min']

	chance = pd.merge(user_is_cancel_min, user_onb_max, on='msno', how='left')
	chance['is_cancel_chance_n5'] = chance.onb_max - chance.onb_min +1

	tbl = pd.merge(cnt, chance, on= ['msno', 'is_cancel'], how='left')
	tbl['cancel_ratio_by_chance_n5'] = tbl.cnt / tbl.is_cancel_chance_n5
	# total_count
	tbl_ = df_.groupby('msno').is_cancel.sum().to_frame()
	tbl_.columns = ['cancel_total_count_n5']
	tbl_['cancel_total_count_ratio_n5'] = df_.groupby('msno').is_cancel.mean()
	tbl_.reset_index(inplace = True)
	tbl = pd.merge(tbl, tbl_, on = 'msno')
	col = ['msno', 'is_cancel_chance_n5', 'cancel_ratio_by_chance_n5','cancel_total_count_n5','cancel_total_count_ratio_n5']
	tbl = tbl[col]
	tbl.drop_duplicates('msno',keep = 'last',inplace = True) # 只要is_cancel == 1
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/is_cancel_n5.csv'.format(folder), index = False)

	del tbl
	del tbl_
	del df_
	gc.collect()
	##################################################
	# only one prvious order
	##################################################
	df_ = df.groupby('msno').apply(near,1).reset_index(drop = True)
	df_ = df_.groupby('msno').apply(make_order_number)
	#count
	cnt = df_.groupby(['msno', 'is_cancel']).size()
	cnt.name = 'cnt'
	cnt = cnt.reset_index()
	# chance
	user_onb_max = df_.groupby('msno').order_number.max().reset_index()
	user_onb_max.columns = ['msno', 'onb_max']
	user_is_cancel_min = df_.groupby(['msno', 'is_cancel']).order_number.min().reset_index()
	user_is_cancel_min.columns = ['msno', 'is_cancel', 'onb_min']

	chance = pd.merge(user_is_cancel_min, user_onb_max, on='msno', how='left')
	chance['is_cancel_chance_n1'] = chance.onb_max - chance.onb_min +1

	tbl = pd.merge(cnt, chance, on= ['msno', 'is_cancel'], how='left')
	tbl['cancel_ratio_by_chance_n1'] = tbl.cnt / tbl.is_cancel_chance_n1
	# total_count
	tbl_ = df_.groupby('msno').is_cancel.sum().to_frame()
	tbl_.columns = ['cancel_total_count_n1']
	tbl_['cancel_total_count_ratio_n1'] = df_.groupby('msno').is_cancel.mean()
	tbl_.reset_index(inplace = True)
	tbl = pd.merge(tbl, tbl_, on = 'msno')
	col = ['msno', 'is_cancel_chance_n1', 'cancel_ratio_by_chance_n1','cancel_total_count_n1','cancel_total_count_ratio_n1']
	tbl = tbl[col]
	tbl.drop_duplicates('msno',keep = 'last',inplace = True) # 只要is_cancel == 1
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/is_cancel_n1.csv'.format(folder), index = False)

	gc.collect()

##################################################
# Main
##################################################
s = time.time()
# make(0)
# make(1)
# make(2)
# make(-1)

mp_pool = mp.Pool(4) # going into the pool, python will do multi processing automatically.
mp_pool.map(make, [0,1,2,-1])

e = time.time()
print (e-s)
