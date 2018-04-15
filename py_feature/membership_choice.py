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

input_col = ['msno','w','is_subscribe_early','do_change_payment_method','do_spend_more_money', 
    'do_extend_payment_days', 'do_paid_more']

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
	#df = df.head( n= 1000)
	#core1
	tbl = df.groupby('msno').is_subscribe_early.sum().to_frame()
	tbl.columns = ['is_subscribe_early_count']
	tbl['is_subscribe_early_ratio'] = df.groupby('msno').is_subscribe_early.mean()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/is_subscribe_early.csv'.format(folder), index = False)

	#core2
	tbl = df.groupby('msno').do_change_payment_method.sum().to_frame()
	tbl.columns = ['do_change_payment_method_count']
	tbl['do_change_payment_method_ratio'] = df.groupby('msno').do_change_payment_method.mean()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_change_payment_method.csv'.format(folder), index = False)


	#core3
	tbl = df.groupby('msno').do_spend_more_money.mean().to_frame()
	tbl.columns = ['do_spend_more_money-mean']
	tbl['do_spend_more_money-min'] = df.groupby('msno').do_spend_more_money.min()
	tbl['do_spend_more_money-max'] = df.groupby('msno').do_spend_more_money.max()
	tbl['do_spend_more_money-median'] = df.groupby('msno').do_spend_more_money.median()
	tbl['do_spend_more_money-std'] = df.groupby('msno').do_spend_more_money.std()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_spend_more_money.csv'.format(folder), index = False)

	#core4
	tbl = df.groupby('msno').do_extend_payment_days.mean().to_frame()
	tbl.columns = ['do_extend_payment_days-mean']
	tbl['do_extend_payment_days-min'] = df.groupby('msno').do_extend_payment_days.min()
	tbl['do_extend_payment_days-max'] = df.groupby('msno').do_extend_payment_days.max()
	tbl['do_extend_payment_days-median'] = df.groupby('msno').do_extend_payment_days.median()
	tbl['do_extend_payment_days-std'] = df.groupby('msno').do_extend_payment_days.std()
	tbl.reset_index(inplace = True)	

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_extend_payment_days.csv'.format(folder), index = False)

	#core5
	tbl = df.groupby('msno').do_paid_more.sum().to_frame()
	tbl.columns = ['do_paid_more_count']
	tbl['do_paid_more_ratio'] = df.groupby('msno').do_paid_more.mean()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_paid_more.csv'.format(folder), index = False)

	##################################################
	# near 5
	##################################################
	df_ = df.groupby('msno').apply(near,5).reset_index(drop = True)

	#core1
	tbl = df_.groupby('msno').is_subscribe_early.sum().to_frame()
	tbl.columns = ['is_subscribe_early_count_n5']
	tbl['is_subscribe_early_ratio_n5'] = df_.groupby('msno').is_subscribe_early.mean()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/is_subscribe_early_n5.csv'.format(folder), index = False)

	#core2
	tbl = df_.groupby('msno').do_change_payment_method.sum().to_frame()
	tbl.columns = ['do_change_payment_method_count_n5']
	tbl['do_change_payment_method_ratio_n5'] = df_.groupby('msno').do_change_payment_method.mean()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_change_payment_method_n5.csv'.format(folder), index = False)


	#core3
	tbl = df_.groupby('msno').do_spend_more_money.mean().to_frame()
	tbl.columns = ['do_spend_more_money-mean_n5']
	tbl['do_spend_more_money-min_n5'] = df_.groupby('msno').do_spend_more_money.min()
	tbl['do_spend_more_money-max_n5'] = df_.groupby('msno').do_spend_more_money.max()
	tbl['do_spend_more_money-median_n5'] = df_.groupby('msno').do_spend_more_money.median()
	tbl['do_spend_more_money-std_n5'] = df_.groupby('msno').do_spend_more_money.std()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_spend_more_money_n5.csv'.format(folder), index = False)

	#core4
	tbl = df_.groupby('msno').do_extend_payment_days.mean().to_frame()
	tbl.columns = ['do_extend_payment_days-mean_n5']
	tbl['do_extend_payment_days-min_n5'] = df_.groupby('msno').do_extend_payment_days.min()
	tbl['do_extend_payment_days-max_n5'] = df_.groupby('msno').do_extend_payment_days.max()
	tbl['do_extend_payment_days-median_n5'] = df_.groupby('msno').do_extend_payment_days.median()
	tbl['do_extend_payment_days-std_n5'] = df_.groupby('msno').do_extend_payment_days.std()
	tbl.reset_index(inplace = True)	

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_extend_payment_days_n5.csv'.format(folder), index = False)

	#core5
	tbl = df_.groupby('msno').do_paid_more.sum().to_frame()
	tbl.columns = ['do_paid_more_count_n5']
	tbl['do_paid_more_ratio_n5'] = df_.groupby('msno').do_paid_more.mean()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_paid_more_n5.csv'.format(folder), index = False)

	del df_
	gc.collect()

	##################################################
	# only one prvious order
	##################################################
	df_ = df.groupby('msno').apply(near,1).reset_index(drop = True)

	#core1
	tbl = df_.groupby('msno').is_subscribe_early.sum().to_frame()
	tbl.columns = ['is_subscribe_early_count_n1']
	tbl['is_subscribe_early_ratio_n1'] = df_.groupby('msno').is_subscribe_early.mean()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/is_subscribe_early_n1.csv'.format(folder), index = False)

	#core2
	tbl = df_.groupby('msno').do_change_payment_method.sum().to_frame()
	tbl.columns = ['do_change_payment_method_count_n1']
	tbl['do_change_payment_method_ratio_n1'] = df_.groupby('msno').do_change_payment_method.mean()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_change_payment_method_n1.csv'.format(folder), index = False)


	#core3
	tbl = df_.groupby('msno').do_spend_more_money.mean().to_frame()
	tbl.columns = ['do_spend_more_money-mean_n1']
	tbl['do_spend_more_money-min_n1'] = df_.groupby('msno').do_spend_more_money.min()
	tbl['do_spend_more_money-max_n1'] = df_.groupby('msno').do_spend_more_money.max()
	tbl['do_spend_more_money-median_n1'] = df_.groupby('msno').do_spend_more_money.median()
	tbl['do_spend_more_money-std_n1'] = df_.groupby('msno').do_spend_more_money.std()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_spend_more_money_n1.csv'.format(folder), index = False)

	#core4
	tbl = df_.groupby('msno').do_extend_payment_days.mean().to_frame()
	tbl.columns = ['do_extend_payment_days-mean_n1']
	tbl['do_extend_payment_days-min_n1'] = df_.groupby('msno').do_extend_payment_days.min()
	tbl['do_extend_payment_days-max_n1'] = df_.groupby('msno').do_extend_payment_days.max()
	tbl['do_extend_payment_days-median_n1'] = df_.groupby('msno').do_extend_payment_days.median()
	tbl['do_extend_payment_days-std_n1'] = df_.groupby('msno').do_extend_payment_days.std()
	tbl.reset_index(inplace = True)	

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_extend_payment_days_n1.csv'.format(folder), index = False)

	#core5
	tbl = df_.groupby('msno').do_paid_more.sum().to_frame()
	tbl.columns = ['do_paid_more_count_n1']
	tbl['do_paid_more_ratio_n1'] = df_.groupby('msno').do_paid_more.mean()
	tbl.reset_index(inplace = True)

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	#write
	tbl.to_csv('../feature/{}/do_paid_more_n1.csv'.format(folder), index = False)

	del df_
	gc.collect()


##################################################
# Main
##################################################
s = time.time()

mp_pool = mp.Pool(4) # going into the pool, python will do multi processing automatically.
mp_pool.map(make, [0,1,2,-1])

e = time.time()
print (e-s)




