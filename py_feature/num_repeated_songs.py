#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

# it took 5.374 hours
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

def within_n_days(x, T, n = 7):
	# n = 7, 14, 30, 60, 90,
	##################################################
	# Filtering accroding to w
	##################################################
	if T == 0:
		# w = 0:使用3月之前的資料當作history
		prediction_deadline = datetime.strptime('2017-03-01', '%Y-%m-%d')
	elif T == 1:
		# w = 1:使用2月之前的資料當作history
		prediction_deadline = datetime.strptime('2017-02-01', '%Y-%m-%d')
	elif T == 2:
		# w = 2:使用1月之前的資料當作history
		prediction_deadline = datetime.strptime('2017-01-01', '%Y-%m-%d')
	elif T == -1:
		# w = -1:使用4月之前的資料當作history
		prediction_deadline = datetime.strptime('2017-04-01', '%Y-%m-%d')
	# n = 7, 14, 21, 30
	check_points = prediction_deadline - timedelta(n) # 往prediction_deadline前n天內
	x = x[ (x.date >= check_points )]
	# 如果有些使用者在預測deadline前n天內沒有資料,到時候再concat, merge的時候會是nan就把它補-1
	return x

#==============================================================================
# def
#==============================================================================
def make(T):
	"""
	T = 0
	folder = 'trainW-0'
	"""
	input_col = ['msno', 'date', 'num_25', 'num_50','num_75', 'num_985', 'num_100','num_unq']

	if T == -1:
		folder = 'test'
		#label
		train = pd.read_csv('../input/sample_submission_v2.csv')[['msno']] # 此train代表的是test的user
		#file1
		user_logs = utils.read_multiple_csv('../feature/{}/compressed_user_logs'.format(folder),
			input_col,
			parse_dates = ['date']) 
		user_logs = pd.concat([user_logs,pd.read_csv('../input/user_logs_v2.csv',parse_dates = ['date'])[input_col]],
		ignore_index = True) 
		#user_logs.sort_values(by = ['msno', 'date'],inplace = True)
	else:
		folder = 'trainW-'+ str(T)
		#label
		train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[['msno']] 
		#file1
		user_logs = utils.read_multiple_csv('../feature/{}/compressed_user_logs'.format(folder), 
			input_col,
			parse_dates = ['date']
			)

	##################################################
	# basic procedure
	##################################################
	#incompleted vs completed 
	user_logs['completed_songs_ratio'] = (user_logs.num_100 + user_logs.num_985)/ (
		user_logs.num_25 + user_logs.num_50 + user_logs.num_75 + user_logs.num_100 + user_logs.num_985)
	#num_repeated_songs
	user_logs['num_repeated_songs'] = (user_logs.num_100 + user_logs.num_985 + user_logs.num_75) / user_logs.num_unq

	user_logs.drop(['num_25','num_50','num_75','num_985','num_100','num_unq'], axis = 1, inplace = True)
	user_logs.dropna(inplace = True) # 把0/0 給drop掉
	gc.collect() # release memory fisrt
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(user_logs)
	df = pd.merge(train, user_logs, on = 'msno', how = 'left')
	del user_logs
	gc.collect()
	print ('shape of df:', df.shape)
	##################################################
	# All history
	##################################################
	#core1:completed_songs_ratio
	print ('core1')
	tbl = df.groupby('msno').completed_songs_ratio.mean().to_frame()
	tbl.columns = ['completed_songs_ratio-mean']
	tbl['completed_songs_ratio-min'] = df.groupby('msno').completed_songs_ratio.min()
	tbl['completed_songs_ratio-max'] = df.groupby('msno').completed_songs_ratio.max()
	tbl['completed_songs_ratio-median'] = df.groupby('msno').completed_songs_ratio.median()
	tbl['completed_songs_ratio-std'] = df.groupby('msno').completed_songs_ratio.std()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/completed_songs_ratio.csv'.format(folder), index = False)
	del tbl
	gc.collect()

	#core2:num_repeated_songs
	print ('core2')
	tbl = df.groupby('msno').num_repeated_songs.mean().to_frame()
	tbl.columns = ['num_repeated_songs-mean']
	tbl['num_repeated_songs-min'] = df.groupby('msno').num_repeated_songs.min()
	tbl['num_repeated_songs-max'] = df.groupby('msno').num_repeated_songs.max()
	tbl['num_repeated_songs-median'] = df.groupby('msno').num_repeated_songs.median()
	tbl['num_repeated_songs-std'] = df.groupby('msno').num_repeated_songs.std()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_repeated_songs.csv'.format(folder), index = False)
	del tbl
	gc.collect()

	##################################################
	# n = 7
	##################################################
	df_ = df.groupby('msno').apply(within_n_days,T, n = 7).reset_index(drop = True)
	#core1:completed_songs_ratio
	print ('core1')
	tbl = df_.groupby('msno').completed_songs_ratio.mean().to_frame()
	tbl.columns = ['completed_songs_ratio_during_t_7-mean']
	tbl['completed_songs_ratio_during_t_7-min'] = df_.groupby('msno').completed_songs_ratio.min()
	tbl['completed_songs_ratio_during_t_7-max'] = df_.groupby('msno').completed_songs_ratio.max()
	tbl['completed_songs_ratio_during_t_7-median'] = df_.groupby('msno').completed_songs_ratio.median()
	tbl['completed_songs_ratio_during_t_7-std'] = df_.groupby('msno').completed_songs_ratio.std()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/completed_songs_ratio_during_t_7.csv'.format(folder), index = False)
	del tbl
	gc.collect()	
	#core2:num_repeated_songs
	print ('core2')
	tbl = df_.groupby('msno').num_repeated_songs.mean().to_frame()
	tbl.columns = ['num_repeated_songs_during_t_7-mean']
	tbl['num_repeated_songs_during_t_7-min'] = df_.groupby('msno').num_repeated_songs.min()
	tbl['num_repeated_songs_during_t_7-max'] = df_.groupby('msno').num_repeated_songs.max()
	tbl['num_repeated_songs_during_t_7-median'] = df_.groupby('msno').num_repeated_songs.median()
	tbl['num_repeated_songs_during_t_7-std'] = df_.groupby('msno').num_repeated_songs.std()
	tbl.reset_index(inplace = True)
	del df_
	gc.collect()
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_repeated_songs_during_t_7.csv'.format(folder), index = False)
	del tbl
	gc.collect()

	##################################################
	# n = 14
	##################################################
	df_ = df.groupby('msno').apply(within_n_days,T, n = 14).reset_index(drop = True)
	#core1:completed_songs_ratio
	print ('core1')
	tbl = df_.groupby('msno').completed_songs_ratio.mean().to_frame()
	tbl.columns = ['completed_songs_ratio_during_t_14-mean']
	tbl['completed_songs_ratio_during_t_14-min'] = df_.groupby('msno').completed_songs_ratio.min()
	tbl['completed_songs_ratio_during_t_14-max'] = df_.groupby('msno').completed_songs_ratio.max()
	tbl['completed_songs_ratio_during_t_14-median'] = df_.groupby('msno').completed_songs_ratio.median()
	tbl['completed_songs_ratio_during_t_14-std'] = df_.groupby('msno').completed_songs_ratio.std()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/completed_songs_ratio_during_t_14.csv'.format(folder), index = False)
	del tbl
	gc.collect()	
	#core2:num_repeated_songs
	print ('core2')
	tbl = df_.groupby('msno').num_repeated_songs.mean().to_frame()
	tbl.columns = ['num_repeated_songs_during_t_14-mean']
	tbl['num_repeated_songs_during_t_14-min'] = df_.groupby('msno').num_repeated_songs.min()
	tbl['num_repeated_songs_during_t_14-max'] = df_.groupby('msno').num_repeated_songs.max()
	tbl['num_repeated_songs_during_t_14-median'] = df_.groupby('msno').num_repeated_songs.median()
	tbl['num_repeated_songs_during_t_14-std'] = df_.groupby('msno').num_repeated_songs.std()
	tbl.reset_index(inplace = True)
	del df_
	gc.collect()
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_repeated_songs_during_t_14.csv'.format(folder), index = False)
	del tbl
	gc.collect()

	##################################################
	# n = 30
	##################################################
	df_ = df.groupby('msno').apply(within_n_days,T, n = 30).reset_index(drop = True)
	#core1:completed_songs_ratio
	print ('core1')
	tbl = df_.groupby('msno').completed_songs_ratio.mean().to_frame()
	tbl.columns = ['completed_songs_ratio_during_t_30-mean']
	tbl['completed_songs_ratio_during_t_30-min'] = df_.groupby('msno').completed_songs_ratio.min()
	tbl['completed_songs_ratio_during_t_30-max'] = df_.groupby('msno').completed_songs_ratio.max()
	tbl['completed_songs_ratio_during_t_30-median'] = df_.groupby('msno').completed_songs_ratio.median()
	tbl['completed_songs_ratio_during_t_30-std'] = df_.groupby('msno').completed_songs_ratio.std()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/completed_songs_ratio_during_t_30.csv'.format(folder), index = False)
	del tbl
	gc.collect()	
	#core2:num_repeated_songs
	print ('core2')
	tbl = df_.groupby('msno').num_repeated_songs.mean().to_frame()
	tbl.columns = ['num_repeated_songs_during_t_30-mean']
	tbl['num_repeated_songs_during_t_30-min'] = df_.groupby('msno').num_repeated_songs.min()
	tbl['num_repeated_songs_during_t_30-max'] = df_.groupby('msno').num_repeated_songs.max()
	tbl['num_repeated_songs_during_t_30-median'] = df_.groupby('msno').num_repeated_songs.median()
	tbl['num_repeated_songs_during_t_30-std'] = df_.groupby('msno').num_repeated_songs.std()
	tbl.reset_index(inplace = True)
	del df_
	gc.collect()
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_repeated_songs_during_t_30.csv'.format(folder), index = False)
	del tbl
	gc.collect()
	##################################################
	# n = 60
	##################################################
	df_ = df.groupby('msno').apply(within_n_days,T, n = 60).reset_index(drop = True)
	#core1:completed_songs_ratio
	print ('core1')
	tbl = df_.groupby('msno').completed_songs_ratio.mean().to_frame()
	tbl.columns = ['completed_songs_ratio_during_t_60-mean']
	tbl['completed_songs_ratio_during_t_60-min'] = df_.groupby('msno').completed_songs_ratio.min()
	tbl['completed_songs_ratio_during_t_60-max'] = df_.groupby('msno').completed_songs_ratio.max()
	tbl['completed_songs_ratio_during_t_60-median'] = df_.groupby('msno').completed_songs_ratio.median()
	tbl['completed_songs_ratio_during_t_60-std'] = df_.groupby('msno').completed_songs_ratio.std()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/completed_songs_ratio_during_t_60.csv'.format(folder), index = False)
	del tbl
	gc.collect()	
	#core2:num_repeated_songs
	print ('core2')
	tbl = df_.groupby('msno').num_repeated_songs.mean().to_frame()
	tbl.columns = ['num_repeated_songs_during_t_60-mean']
	tbl['num_repeated_songs_during_t_60-min'] = df_.groupby('msno').num_repeated_songs.min()
	tbl['num_repeated_songs_during_t_60-max'] = df_.groupby('msno').num_repeated_songs.max()
	tbl['num_repeated_songs_during_t_60-median'] = df_.groupby('msno').num_repeated_songs.median()
	tbl['num_repeated_songs_during_t_60-std'] = df_.groupby('msno').num_repeated_songs.std()
	tbl.reset_index(inplace = True)
	del df_
	gc.collect()
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_repeated_songs_during_t_60.csv'.format(folder), index = False)
	del tbl
	gc.collect()
	##################################################
	# n = 90
	##################################################
	df_ = df.groupby('msno').apply(within_n_days,T, n = 90).reset_index(drop = True)
	#core1:completed_songs_ratio
	print ('core1')
	tbl = df_.groupby('msno').completed_songs_ratio.mean().to_frame()
	tbl.columns = ['completed_songs_ratio_during_t_90-mean']
	tbl['completed_songs_ratio_during_t_90-min'] = df_.groupby('msno').completed_songs_ratio.min()
	tbl['completed_songs_ratio_during_t_90-max'] = df_.groupby('msno').completed_songs_ratio.max()
	tbl['completed_songs_ratio_during_t_90-median'] = df_.groupby('msno').completed_songs_ratio.median()
	tbl['completed_songs_ratio_during_t_90-std'] = df_.groupby('msno').completed_songs_ratio.std()
	tbl.reset_index(inplace = True)
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/completed_songs_ratio_during_t_90.csv'.format(folder), index = False)
	del tbl
	gc.collect()	
	#core2:num_repeated_songs
	print ('core2')
	tbl = df_.groupby('msno').num_repeated_songs.mean().to_frame()
	tbl.columns = ['num_repeated_songs_during_t_90-mean']
	tbl['num_repeated_songs_during_t_90-min'] = df_.groupby('msno').num_repeated_songs.min()
	tbl['num_repeated_songs_during_t_90-max'] = df_.groupby('msno').num_repeated_songs.max()
	tbl['num_repeated_songs_during_t_90-median'] = df_.groupby('msno').num_repeated_songs.median()
	tbl['num_repeated_songs_during_t_90-std'] = df_.groupby('msno').num_repeated_songs.std()
	tbl.reset_index(inplace = True)
	del df_
	gc.collect()
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/num_repeated_songs_during_t_90.csv'.format(folder), index = False)
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

e = time.time()
print (e-s)








