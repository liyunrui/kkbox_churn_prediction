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
input_col = ['msno','w','days_since_the_last_expiration',
'days_since_the_last_subscription','days_since_the_last_expiration-cumsum',
'days_since_the_last_expiration_ratio','days_since_the_last_subscription_ratio',
'days_since_the_first_subscription']

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

	if T == -1:
		folder = 'test'
		train = pd.read_csv('../input/sample_submission_v2.csv') # 此train代表的是test的user
		train['w'] = T
		membership_loyalty = utils.read_multiple_csv('../feature/{}/days_since_the_last_transactions'.format(folder)
			,input_col) 

	else:
		folder = 'trainW-'+str(T)
		membership_loyalty = utils.read_multiple_csv('../feature/{}/days_since_the_last_transactions'.format(folder)
			,input_col) 
		train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[['msno','w']] # we do not need is_churn
		#==============================================================================
		print('reduce memory')
		#==============================================================================
		utils.reduce_memory(membership_loyalty)
	##################################################
	# All history
	##################################################
	# merge
	df = pd.merge(train, 
	    membership_loyalty, 
	    on=['msno','w'], 
	    how='left')
	########
	# core1
	########
	tbl = df.groupby('msno').days_since_the_last_expiration.mean().to_frame()
	tbl.columns = ['days_since_the_last_expiration-mean']
	tbl['days_since_the_last_expiration-min'] = df.groupby('msno').days_since_the_last_expiration.min()
	tbl['days_since_the_last_expiration-max'] = df.groupby('msno').days_since_the_last_expiration.max()
	tbl['days_since_the_last_expiration-median'] = df.groupby('msno').days_since_the_last_expiration.median()
	tbl['days_since_the_last_expiration-std'] = df.groupby('msno').days_since_the_last_expiration.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)	
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_expiration.csv'.format(folder), index = False)
	########
	# core2
	########
	tbl = df.groupby('msno').days_since_the_last_subscription.mean().to_frame()
	tbl.columns = ['days_since_the_last_subscription-mean']
	tbl['days_since_the_last_subscription-min'] = df.groupby('msno').days_since_the_last_subscription.min()
	tbl['days_since_the_last_subscription-max'] = df.groupby('msno').days_since_the_last_subscription.max()
	tbl['days_since_the_last_subscription-median'] = df.groupby('msno').days_since_the_last_subscription.median()
	tbl['days_since_the_last_subscription-std'] = df.groupby('msno').days_since_the_last_subscription.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_subscription.csv'.format(folder), index = False)
	#########
	# core3
	#########
	tbl = df.groupby('msno')['days_since_the_last_expiration-cumsum'].mean().to_frame()
	tbl.columns = ['days_since_the_last_expiration-cumsum-mean']
	tbl['days_since_the_last_expiration-cumsum-min'] = df.groupby('msno')['days_since_the_last_expiration-cumsum'].min()
	tbl['days_since_the_last_expiration-cumsum-max'] = df.groupby('msno')['days_since_the_last_expiration-cumsum'].max()
	tbl['days_since_the_last_expiration-cumsum-median'] = df.groupby('msno')['days_since_the_last_expiration-cumsum'].median()
	tbl['days_since_the_last_expiration-cumsum-std'] = df.groupby('msno')['days_since_the_last_expiration-cumsum'].std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_expiration-cumsum.csv'.format(folder), index = False)
	########
	# core4
	########
	tbl = df.groupby('msno').days_since_the_last_expiration_ratio.mean().to_frame()
	tbl.columns = ['days_since_the_last_expiration_ratio-mean']
	tbl['days_since_the_last_expiration_ratio-min'] = df.groupby('msno').days_since_the_last_expiration_ratio.min()
	tbl['days_since_the_last_expiration_ratio-max'] = df.groupby('msno').days_since_the_last_expiration_ratio.max()
	tbl['days_since_the_last_expiration_ratio-median'] = df.groupby('msno').days_since_the_last_expiration_ratio.median()
	tbl['days_since_the_last_expiration_ratio-std'] = df.groupby('msno').days_since_the_last_expiration_ratio.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_expiration_ratio.csv'.format(folder), index = False)
	########
	# core5
	########
	tbl = df.groupby('msno').days_since_the_last_subscription_ratio.mean().to_frame()
	tbl.columns = ['days_since_the_last_subscription_ratio-mean']
	tbl['days_since_the_last_subscription_ratio-min'] = df.groupby('msno').days_since_the_last_subscription_ratio.min()
	tbl['days_since_the_last_subscription_ratio-max'] = df.groupby('msno').days_since_the_last_subscription_ratio.max()
	tbl['days_since_the_last_subscription_ratio-median'] = df.groupby('msno').days_since_the_last_subscription_ratio.median()
	tbl['days_since_the_last_subscription_ratio-std'] = df.groupby('msno').days_since_the_last_subscription_ratio.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_subscription_ratio.csv'.format(folder), index = False)
	########
	# core6
	########
	tbl = df.groupby('msno').days_since_the_first_subscription.mean().to_frame()
	tbl.columns = ['days_since_the_first_subscription-mean']
	tbl['days_since_the_first_subscription-min'] = df.groupby('msno').days_since_the_first_subscription.min()
	tbl['days_since_the_first_subscription-max'] = df.groupby('msno').days_since_the_first_subscription.max()
	tbl['days_since_the_first_subscription-median'] = df.groupby('msno').days_since_the_first_subscription.median()
	tbl['days_since_the_first_subscription-std'] = df.groupby('msno').days_since_the_first_subscription.std()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)	
	# write
	tbl.to_csv('../feature/{}/days_since_the_first_subscription.csv'.format(folder), index = False)


	##################################################
	# near 5
	##################################################
	df_ = df.groupby('msno').apply(near,5).reset_index(drop = True)
	########
	# core1
	########
	tbl = df_.groupby('msno').days_since_the_last_expiration.mean().to_frame()
	tbl.columns = ['days_since_the_last_expiration-mean_n5']
	tbl['days_since_the_last_expiration-min_n5'] = df_.groupby('msno').days_since_the_last_expiration.min()
	tbl['days_since_the_last_expiration-max_n5'] = df_.groupby('msno').days_since_the_last_expiration.max()
	tbl['days_since_the_last_expiration-median_n5'] = df_.groupby('msno').days_since_the_last_expiration.median()
	tbl['days_since_the_last_expiration-std_n5'] = df_.groupby('msno').days_since_the_last_expiration.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/days_since_the_last_expiration_n5.csv'.format(folder), index = False)
	########
	# core2
	########
	tbl = df_.groupby('msno').days_since_the_last_subscription.mean().to_frame()
	tbl.columns = ['days_since_the_last_subscription-mean_n5']
	tbl['days_since_the_last_subscription-min_n5'] = df_.groupby('msno').days_since_the_last_subscription.min()
	tbl['days_since_the_last_subscription-max_n5'] = df_.groupby('msno').days_since_the_last_subscription.max()
	tbl['days_since_the_last_subscription-median_n5'] = df_.groupby('msno').days_since_the_last_subscription.median()
	tbl['days_since_the_last_subscription-std_n5'] = df_.groupby('msno').days_since_the_last_subscription.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_subscription_n5.csv'.format(folder), index = False)
	#########
	# core3
	#########
	tbl = df_.groupby('msno')['days_since_the_last_expiration-cumsum'].mean().to_frame()
	tbl.columns = ['days_since_the_last_expiration-cumsum-mean_n5']
	tbl['days_since_the_last_expiration-cumsum-min_n5'] = df_.groupby('msno')['days_since_the_last_expiration-cumsum'].min()
	tbl['days_since_the_last_expiration-cumsum-max_n5'] = df_.groupby('msno')['days_since_the_last_expiration-cumsum'].max()
	tbl['days_since_the_last_expiration-cumsum-median_n5'] = df_.groupby('msno')['days_since_the_last_expiration-cumsum'].median()
	tbl['days_since_the_last_expiration-cumsum-std_n5'] = df_.groupby('msno')['days_since_the_last_expiration-cumsum'].std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_expiration-cumsum_n5.csv'.format(folder), index = False)
	########
	# core4
	########
	tbl = df_.groupby('msno').days_since_the_last_expiration_ratio.mean().to_frame()
	tbl.columns = ['days_since_the_last_expiration_ratio-mean_n5']
	tbl['days_since_the_last_expiration_ratio-min_n5'] = df_.groupby('msno').days_since_the_last_expiration_ratio.min()
	tbl['days_since_the_last_expiration_ratio-max_n5'] = df_.groupby('msno').days_since_the_last_expiration_ratio.max()
	tbl['days_since_the_last_expiration_ratio-median_n5'] = df_.groupby('msno').days_since_the_last_expiration_ratio.median()
	tbl['days_since_the_last_expiration_ratio-std_n5'] = df_.groupby('msno').days_since_the_last_expiration_ratio.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_expiration_ratio_n5.csv'.format(folder), index = False)
	########
	# core5
	########
	tbl = df_.groupby('msno').days_since_the_last_subscription_ratio.mean().to_frame()
	tbl.columns = ['days_since_the_last_subscription_ratio-mean_n5']
	tbl['days_since_the_last_subscription_ratio-min_n5'] = df_.groupby('msno').days_since_the_last_subscription_ratio.min()
	tbl['days_since_the_last_subscription_ratio-max_n5'] = df_.groupby('msno').days_since_the_last_subscription_ratio.max()
	tbl['days_since_the_last_subscription_ratio-median_n5'] = df_.groupby('msno').days_since_the_last_subscription_ratio.median()
	tbl['days_since_the_last_subscription_ratio-std_n5'] = df_.groupby('msno').days_since_the_last_subscription_ratio.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_subscription_ratio_n5.csv'.format(folder), index = False)
	########
	# core6
	########
	tbl = df_.groupby('msno').days_since_the_first_subscription.mean().to_frame()
	tbl.columns = ['days_since_the_first_subscription-mean_n5']
	tbl['days_since_the_first_subscription-min_n5'] = df_.groupby('msno').days_since_the_first_subscription.min()
	tbl['days_since_the_first_subscription-max_n5'] = df_.groupby('msno').days_since_the_first_subscription.max()
	tbl['days_since_the_first_subscription-median_n5'] = df_.groupby('msno').days_since_the_first_subscription.median()
	tbl['days_since_the_first_subscription-std_n5'] = df_.groupby('msno').days_since_the_first_subscription.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_first_subscription_n5.csv'.format(folder), index = False)

	del df_
	##################################################
	# only one prvious order
	##################################################
	df_ = df.groupby('msno').apply(near,1).reset_index(drop = True)
	########
	# core1
	########
	tbl = df_.groupby('msno').days_since_the_last_expiration.mean().to_frame()
	tbl.columns = ['days_since_the_last_expiration-mean_n1']
	tbl['days_since_the_last_expiration-min_n1'] = df_.groupby('msno').days_since_the_last_expiration.min()
	tbl['days_since_the_last_expiration-max_n1'] = df_.groupby('msno').days_since_the_last_expiration.max()
	tbl['days_since_the_last_expiration-median_n1'] = df_.groupby('msno').days_since_the_last_expiration.median()
	tbl['days_since_the_last_expiration-std_n1'] = df_.groupby('msno').days_since_the_last_expiration.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_expiration_n1.csv'.format(folder), index = False)
	########
	# core2
	########
	tbl = df_.groupby('msno').days_since_the_last_subscription.mean().to_frame()
	tbl.columns = ['days_since_the_last_subscription-mean_n1']
	tbl['days_since_the_last_subscription-min_n1'] = df_.groupby('msno').days_since_the_last_subscription.min()
	tbl['days_since_the_last_subscription-max_n1'] = df_.groupby('msno').days_since_the_last_subscription.max()
	tbl['days_since_the_last_subscription-median_n1'] = df_.groupby('msno').days_since_the_last_subscription.median()
	tbl['days_since_the_last_subscription-std_n1'] = df_.groupby('msno').days_since_the_last_subscription.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_subscription_n1.csv'.format(folder), index = False)
	#########
	# core3
	#########
	tbl = df_.groupby('msno')['days_since_the_last_expiration-cumsum'].mean().to_frame()
	tbl.columns = ['days_since_the_last_expiration-cumsum-mean_n1']
	tbl['days_since_the_last_expiration-cumsum-min_n1'] = df_.groupby('msno')['days_since_the_last_expiration-cumsum'].min()
	tbl['days_since_the_last_expiration-cumsum-max_n1'] = df_.groupby('msno')['days_since_the_last_expiration-cumsum'].max()
	tbl['days_since_the_last_expiration-cumsum-median_n1'] = df_.groupby('msno')['days_since_the_last_expiration-cumsum'].median()
	tbl['days_since_the_last_expiration-cumsum-std_n1'] = df_.groupby('msno')['days_since_the_last_expiration-cumsum'].std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_expiration-cumsum_n1.csv'.format(folder), index = False)
	########
	# core4
	########
	tbl = df_.groupby('msno').days_since_the_last_expiration_ratio.mean().to_frame()
	tbl.columns = ['days_since_the_last_expiration_ratio-mean_n1']
	tbl['days_since_the_last_expiration_ratio-min_n1'] = df_.groupby('msno').days_since_the_last_expiration_ratio.min()
	tbl['days_since_the_last_expiration_ratio-max_n1'] = df_.groupby('msno').days_since_the_last_expiration_ratio.max()
	tbl['days_since_the_last_expiration_ratio-median_n1'] = df_.groupby('msno').days_since_the_last_expiration_ratio.median()
	tbl['days_since_the_last_expiration_ratio-std_n1'] = df_.groupby('msno').days_since_the_last_expiration_ratio.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_expiration_ratio_n1.csv'.format(folder), index = False)
	########
	# core5
	########
	tbl = df_.groupby('msno').days_since_the_last_subscription_ratio.mean().to_frame()
	tbl.columns = ['days_since_the_last_subscription_ratio-mean_n1']
	tbl['days_since_the_last_subscription_ratio-min_n1'] = df_.groupby('msno').days_since_the_last_subscription_ratio.min()
	tbl['days_since_the_last_subscription_ratio-max_n1'] = df_.groupby('msno').days_since_the_last_subscription_ratio.max()
	tbl['days_since_the_last_subscription_ratio-median_n1'] = df_.groupby('msno').days_since_the_last_subscription_ratio.median()
	tbl['days_since_the_last_subscription_ratio-std_n1'] = df_.groupby('msno').days_since_the_last_subscription_ratio.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_last_subscription_ratio_n1.csv'.format(folder), index = False)
	########
	# core6
	########
	tbl = df_.groupby('msno').days_since_the_first_subscription.mean().to_frame()
	tbl.columns = ['days_since_the_first_subscription-mean_n1']
	tbl['days_since_the_first_subscription-min_n1'] = df_.groupby('msno').days_since_the_first_subscription.min()
	tbl['days_since_the_first_subscription-max_n1'] = df_.groupby('msno').days_since_the_first_subscription.max()
	tbl['days_since_the_first_subscription-median_n1'] = df_.groupby('msno').days_since_the_first_subscription.median()
	tbl['days_since_the_first_subscription-std_n1'] = df_.groupby('msno').days_since_the_first_subscription.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/days_since_the_first_subscription_n1.csv'.format(folder), index = False)

	del df_

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











        
