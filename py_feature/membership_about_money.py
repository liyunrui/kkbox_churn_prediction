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
from collections import Counter

##################################################
# Load  
##################################################
input_col = ['msno', 'transaction_date', 'discount', 'is_discount', 'amt_per_day',
       'cp_value']
transactions_price_plan_days = utils.read_multiple_csv('../input/preprocessed_data/transaction_price_and_play_days_base') # 20,000,000
#transactions_price_plan_days = transactions_price_plan_days.head( n = 1000)
##################################################
# Convert string to datetime format
##################################################
transactions_price_plan_days['transaction_date']  = transactions_price_plan_days.transaction_date.apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d'))

#==============================================================================
print('reduce memory')
#==============================================================================
utils.reduce_memory(transactions_price_plan_days)



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
	    train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[['msno']]

	# the following style is silly, but it's all for saving memory 
	if T == 0:
		df = pd.merge(train, 
	    transactions_price_plan_days[(transactions_price_plan_days.transaction_date < datetime.strptime('2017-03-01', '%Y-%m-%d'))], 
	    on=['msno'], 
	    how='left')
	elif T == 1:
	    # w = 1:使用2月之前的資料當作history
	    df = pd.merge(train, 
	    	transactions_price_plan_days[(transactions_price_plan_days.transaction_date < datetime.strptime('2017-02-01', '%Y-%m-%d'))],
	    	on=['msno'], 
	    	how='left') 
	elif T == 2:
	    # w = 2:使用1月之前的資料當作history
	    df = pd.merge(train, 
	    	transactions_price_plan_days[(transactions_price_plan_days.transaction_date < datetime.strptime('2017-01-01', '%Y-%m-%d'))],
	    	on=['msno'], 
	    	how='left') 
	elif T == -1:
	    # w = -1:使用4月之前的資料當作history
	    df = pd.merge(train, 
	    	transactions_price_plan_days[(transactions_price_plan_days.transaction_date < datetime.strptime('2017-04-01', '%Y-%m-%d'))],
	    	on='msno', 
	    	how='left') 
	
	del train
	gc.collect()
	##################################################
	# All history
	##################################################
	#df = df.dropna()
	########
	# core1
	########
	tbl = df.groupby('msno').discount.mean().to_frame()
	tbl.columns = ['discount-mean']
	tbl['discount-min'] = df.groupby('msno').discount.min()
	tbl['discount-max'] = df.groupby('msno').discount.max()
	tbl['discount-median'] = df.groupby('msno').discount.median()
	tbl['discount-std'] = df.groupby('msno').discount.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/discount.csv'.format(folder), index = False)

	########
	# core2
	########
	tbl = df.groupby('msno').amt_per_day.mean().to_frame()
	tbl.columns = ['amt_per_day-mean']
	tbl['amt_per_day-min'] = df.groupby('msno').amt_per_day.min()
	tbl['amt_per_day-max'] = df.groupby('msno').amt_per_day.max()
	tbl['amt_per_day-median'] = df.groupby('msno').amt_per_day.median()
	tbl['amt_per_day-std'] = df.groupby('msno').amt_per_day.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/amt_per_day.csv'.format(folder), index = False)

	########
	# core3
	########
	tbl = df.groupby('msno').cp_value.mean().to_frame()
	tbl.columns = ['cp_value-mean']
	tbl['cp_value-min'] = df.groupby('msno').cp_value.min()
	tbl['cp_value-max'] = df.groupby('msno').cp_value.max()
	tbl['cp_value-median'] = df.groupby('msno').cp_value.median()
	tbl['cp_value-std'] = df.groupby('msno').cp_value.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/cp_value.csv'.format(folder), index = False)
	########
	# core4
	########
	tbl = df.groupby('msno').is_discount.sum().to_frame()
	tbl.columns = ['is_discount_total_count']
	tbl['is_discount_total_count_ratio'] = df.groupby('msno').is_discount.mean()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/is_discount.csv'.format(folder), index = False)

	##################################################
	# near 5
	##################################################
	df_ = df.groupby('msno').apply(near,5).reset_index(drop = True)
	########
	# core1
	########
	tbl = df_.groupby('msno').discount.mean().to_frame()
	tbl.columns = ['discount-mean_n5']
	tbl['discount-min_n5'] = df_.groupby('msno').discount.min()
	tbl['discount-max_n5'] = df_.groupby('msno').discount.max()
	tbl['discount-median_n5'] = df_.groupby('msno').discount.median()
	tbl['discount-std_n5'] = df_.groupby('msno').discount.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/discount_n5.csv'.format(folder), index = False)

	########
	# core2
	########
	tbl = df_.groupby('msno').amt_per_day.mean().to_frame()
	tbl.columns = ['amt_per_day-mean_n5']
	tbl['amt_per_day-min_n5'] = df_.groupby('msno').amt_per_day.min()
	tbl['amt_per_day-max_n5'] = df_.groupby('msno').amt_per_day.max()
	tbl['amt_per_day-median_n5'] = df_.groupby('msno').amt_per_day.median()
	tbl['amt_per_day-std_n5'] = df_.groupby('msno').amt_per_day.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/amt_per_day_n5.csv'.format(folder), index = False)

	########
	# core3
	########
	tbl = df_.groupby('msno').cp_value.mean().to_frame()
	tbl.columns = ['cp_value-mean_n5']
	tbl['cp_value-min_n5'] = df_.groupby('msno').cp_value.min()
	tbl['cp_value-max_n5'] = df_.groupby('msno').cp_value.max()
	tbl['cp_value-median_n5'] = df_.groupby('msno').cp_value.median()
	tbl['cp_value-std_n5'] = df_.groupby('msno').cp_value.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/cp_value_n5.csv'.format(folder), index = False)
	########
	# core4
	########
	tbl = df_.groupby('msno').is_discount.sum().to_frame()
	tbl.columns = ['is_discount_total_count_n5']
	tbl['is_discount_total_count_ratio_n5'] = df_.groupby('msno').is_discount.mean()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/is_discount_n5.csv'.format(folder), index = False)
	del df_
	gc.collect()

	##################################################
	# only one prvious order
	##################################################
	df_ = df.groupby('msno').apply(near,1).reset_index(drop = True)
	########
	# core1
	########
	tbl = df_.groupby('msno').discount.mean().to_frame()
	tbl.columns = ['discount-mean_n1']
	tbl['discount-min_n1'] = df_.groupby('msno').discount.min()
	tbl['discount-max_n1'] = df_.groupby('msno').discount.max()
	tbl['discount-median_n1'] = df_.groupby('msno').discount.median()
	tbl['discount-std_n1'] = df_.groupby('msno').discount.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/discount_n1.csv'.format(folder), index = False)

	########
	# core2
	########
	tbl = df_.groupby('msno').amt_per_day.mean().to_frame()
	tbl.columns = ['amt_per_day-mean_n1']
	tbl['amt_per_day-min_n1'] = df_.groupby('msno').amt_per_day.min()
	tbl['amt_per_day-max_n1'] = df_.groupby('msno').amt_per_day.max()
	tbl['amt_per_day-median_n1'] = df_.groupby('msno').amt_per_day.median()
	tbl['amt_per_day-std_n1'] = df_.groupby('msno').amt_per_day.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/amt_per_day_n1.csv'.format(folder), index = False)

	########
	# core3
	########
	tbl = df_.groupby('msno').cp_value.mean().to_frame()
	tbl.columns = ['cp_value-mean_n1']
	tbl['cp_value-min_n1'] = df_.groupby('msno').cp_value.min()
	tbl['cp_value-max_n1'] = df_.groupby('msno').cp_value.max()
	tbl['cp_value-median_n1'] = df_.groupby('msno').cp_value.median()
	tbl['cp_value-std_n1'] = df_.groupby('msno').cp_value.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/cp_value_n1.csv'.format(folder), index = False)
	########
	# core4
	########
	tbl = df_.groupby('msno').is_discount.sum().to_frame()
	tbl.columns = ['is_discount_total_count_n1']
	tbl['is_discount_total_count_ratio_n1'] = df_.groupby('msno').is_discount.mean()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/is_discount_n1.csv'.format(folder), index = False)
	del df_
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

