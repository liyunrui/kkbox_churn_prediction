#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

10329
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

input_col = ['msno','num_25','num_50','num_75','num_985','num_100','num_unq']

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
		user_logs = utils.read_multiple_csv('../feature/{}/compressed_user_logs'.format(folder),input_col) 
		user_logs = pd.concat([user_logs,pd.read_csv('../input/user_logs_v2.csv')], ignore_index=True) # user_logs_v2.csv: inclue data of March, it's for testing set.
		user_logs.sort_values(by = ['msno', 'date'],inplace = True)
		train = pd.read_csv('../input/sample_submission_v2.csv') # 此train代表的是test的user
	else:
		folder = 'trainW-'+str(T)
		user_logs = utils.read_multiple_csv('../feature/{}/compressed_user_logs'.format(folder), input_col)
		train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[['msno']] # we do not need is_churn
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(user_logs)
	utils.reduce_memory(train)

	#user_logs = user_logs.head(n = 5000)
	#merge
	df = pd.merge(train,user_logs, on = 'msno', how = 'left')
	#df = df.dropna()	
	del user_logs
	gc.collect()
	##################################################
	# All history
	##################################################

	########
	# core1
	########
	tbl = df.groupby('msno').num_25.mean().to_frame()
	tbl.columns = ['num_25-mean']
	tbl['num_25-min'] = df.groupby('msno').num_25.min()
	tbl['num_25-max'] = df.groupby('msno').num_25.max()
	tbl['num_25-median'] = df.groupby('msno').num_25.median()
	tbl['num_25-std'] = df.groupby('msno').num_25.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/num_25.csv'.format(folder), index = False)
	########
	# core2
	########
	tbl = df.groupby('msno').num_50.mean().to_frame()
	tbl.columns = ['num_50-mean']
	tbl['num_50-min'] = df.groupby('msno').num_50.min()
	tbl['num_50-max'] = df.groupby('msno').num_50.max()
	tbl['num_50-median'] = df.groupby('msno').num_50.median()
	tbl['num_50-std'] = df.groupby('msno').num_50.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/num_50.csv'.format(folder), index = False)
	########
	# core3
	########
	tbl = df.groupby('msno').num_75.mean().to_frame()
	tbl.columns = ['num_75-mean']
	tbl['num_75-min'] = df.groupby('msno').num_75.min()
	tbl['num_75-max'] = df.groupby('msno').num_75.max()
	tbl['num_75-median'] = df.groupby('msno').num_75.median()
	tbl['num_75-std'] = df.groupby('msno').num_75.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/num_75.csv'.format(folder), index = False)
	########
	# core4
	########
	tbl = df.groupby('msno').num_985.mean().to_frame()
	tbl.columns = ['num_985-mean']
	tbl['num_985-min'] = df.groupby('msno').num_985.min()
	tbl['num_985-max'] = df.groupby('msno').num_985.max()
	tbl['num_985-median'] = df.groupby('msno').num_985.median()
	tbl['num_985-std'] = df.groupby('msno').num_985.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/num_985.csv'.format(folder), index = False)
	########
	# core5
	########
	tbl = df.groupby('msno').num_100.mean().to_frame()
	tbl.columns = ['num_100-mean']
	tbl['num_100-min'] = df.groupby('msno').num_100.min()
	tbl['num_100-max'] = df.groupby('msno').num_100.max()
	tbl['num_100-median'] = df.groupby('msno').num_100.median()
	tbl['num_100-std'] = df.groupby('msno').num_100.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/num_100.csv'.format(folder), index = False)
	########
	# core6
	########
	tbl = df.groupby('msno').num_unq.mean().to_frame()
	tbl.columns = ['num_unq-mean']
	tbl['num_unq-min'] = df.groupby('msno').num_unq.min()
	tbl['num_unq-max'] = df.groupby('msno').num_unq.max()
	tbl['num_unq-median'] = df.groupby('msno').num_unq.median()
	tbl['num_unq-std'] = df.groupby('msno').num_unq.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/num_unq.csv'.format(folder), index = False)
	del tbl
	gc.collect()

	##################################################
	# near 5
	##################################################
	df_ = df.groupby('msno').apply(near,5).reset_index(drop = True)
	########
	# core1
	########
	tbl = df_.groupby('msno').num_25.mean().to_frame()
	tbl.columns = ['num_25-mean_n5']
	tbl['num_25-min_n5'] = df_.groupby('msno').num_25.min()
	tbl['num_25-max_n5'] = df_.groupby('msno').num_25.max()
	tbl['num_25-median_n5'] = df_.groupby('msno').num_25.median()
	tbl['num_25-std_n5'] = df_.groupby('msno').num_25.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/num_25_n5.csv'.format(folder), index = False)
	########
	# core2
	########
	tbl = df_.groupby('msno').num_50.mean().to_frame()
	tbl.columns = ['num_50-mean_n5']
	tbl['num_50-min_n5'] = df_.groupby('msno').num_50.min()
	tbl['num_50-max_n5'] = df_.groupby('msno').num_50.max()
	tbl['num_50-median_n5'] = df_.groupby('msno').num_50.median()
	tbl['num_50-std_n5'] = df_.groupby('msno').num_50.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_50_n5.csv'.format(folder), index = False)
	#########
	# core3
	#########
	tbl = df_.groupby('msno').num_75.mean().to_frame()
	tbl.columns = ['num_75-mean_n5']
	tbl['num_75-min_n5'] = df_.groupby('msno').num_75.min()
	tbl['num_75-max_n5'] = df_.groupby('msno').num_75.max()
	tbl['num_75-median_n5'] = df_.groupby('msno').num_75.median()
	tbl['num_75-std_n5'] = df_.groupby('msno').num_75.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_75_n5.csv'.format(folder), index = False)
	########
	# core4
	########
	tbl = df_.groupby('msno').num_985.mean().to_frame()
	tbl.columns = ['num_985-mean_n5']
	tbl['num_985-min_n5'] = df_.groupby('msno').num_985.min()
	tbl['num_985-max_n5'] = df_.groupby('msno').num_985.max()
	tbl['num_985-median_n5'] = df_.groupby('msno').num_985.median()
	tbl['num_985-std_n5'] = df_.groupby('msno').num_985.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_985_n5.csv'.format(folder), index = False)
	########
	# core5
	########
	tbl = df_.groupby('msno').num_100.mean().to_frame()
	tbl.columns = ['num_100-mean_n5']
	tbl['num_100-min_n5'] = df_.groupby('msno').num_100.min()
	tbl['num_100-max_n5'] = df_.groupby('msno').num_100.max()
	tbl['num_100-median_n5'] = df_.groupby('msno').num_100.median()
	tbl['num_100-std_n5'] = df_.groupby('msno').num_100.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_100_n5.csv'.format(folder), index = False)
	########
	# core6
	########
	tbl = df_.groupby('msno').num_unq.mean().to_frame()
	tbl.columns = ['num_unq-mean_n5']
	tbl['num_unq-min_n5'] = df_.groupby('msno').num_unq.min()
	tbl['num_unq-max_n5'] = df_.groupby('msno').num_unq.max()
	tbl['num_unq-median_n5'] = df_.groupby('msno').num_unq.median()
	tbl['num_unq-std_n5'] = df_.groupby('msno').num_unq.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_unq_n5.csv'.format(folder), index = False)
	del tbl
	del df_
	gc.collect()
	##################################################
	# only one prvious order
	##################################################
	df_ = df.groupby('msno').apply(near,1).reset_index(drop = True)
	########
	# core1
	########
	tbl = df_.groupby('msno').num_25.mean().to_frame()
	tbl.columns = ['num_25-mean_n1']
	tbl['num_25-min_n1'] = df_.groupby('msno').num_25.min()
	tbl['num_25-max_n1'] = df_.groupby('msno').num_25.max()
	tbl['num_25-median_n1'] = df_.groupby('msno').num_25.median()
	tbl['num_25-std_n1'] = df_.groupby('msno').num_25.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)

	# write
	tbl.to_csv('../feature/{}/num_25_n1.csv'.format(folder), index = False)
	########
	# core2
	########
	tbl = df_.groupby('msno').num_50.mean().to_frame()
	tbl.columns = ['num_50-mean_n1']
	tbl['num_50-min_n1'] = df_.groupby('msno').num_50.min()
	tbl['num_50-max_n1'] = df_.groupby('msno').num_50.max()
	tbl['num_50-median_n1'] = df_.groupby('msno').num_50.median()
	tbl['num_50-std_n1'] = df_.groupby('msno').num_50.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_50_n1.csv'.format(folder), index = False)
	#########
	# core3
	#########
	tbl = df_.groupby('msno').num_75.mean().to_frame()
	tbl.columns = ['num_75-mean_n1']
	tbl['num_75-min_n1'] = df_.groupby('msno').num_75.min()
	tbl['num_75-max_n1'] = df_.groupby('msno').num_75.max()
	tbl['num_75-median_n1'] = df_.groupby('msno').num_75.median()
	tbl['num_75-std_n1'] = df_.groupby('msno').num_75.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_75_n1.csv'.format(folder), index = False)
	########
	# core4
	########
	tbl = df_.groupby('msno').num_985.mean().to_frame()
	tbl.columns = ['num_985-mean_n1']
	tbl['num_985-min_n1'] = df_.groupby('msno').num_985.min()
	tbl['num_985-max_n1'] = df_.groupby('msno').num_985.max()
	tbl['num_985-median_n1'] = df_.groupby('msno').num_985.median()
	tbl['num_985-std_n1'] = df_.groupby('msno').num_985.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_985_n1.csv'.format(folder), index = False)
	########
	# core5
	########
	tbl = df_.groupby('msno').num_100.mean().to_frame()
	tbl.columns = ['num_100-mean_n1']
	tbl['num_100-min_n1'] = df_.groupby('msno').num_100.min()
	tbl['num_100-max_n1'] = df_.groupby('msno').num_100.max()
	tbl['num_100-median_n1'] = df_.groupby('msno').num_100.median()
	tbl['num_100-std_n1'] = df_.groupby('msno').num_100.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_100_n1.csv'.format(folder), index = False)
	########
	# core6
	########
	tbl = df_.groupby('msno').num_unq.mean().to_frame()
	tbl.columns = ['num_unq-mean_n1']
	tbl['num_unq-min_n1'] = df_.groupby('msno').num_unq.min()
	tbl['num_unq-max_n1'] = df_.groupby('msno').num_unq.max()
	tbl['num_unq-median_n1'] = df_.groupby('msno').num_unq.median()
	tbl['num_unq-std_n1'] = df_.groupby('msno').num_unq.std()
	tbl.reset_index(inplace = True)	
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_unq_n1.csv'.format(folder), index = False)
	del tbl
	del df_
	gc.collect()


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










