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


def drop_first_columns(x):
    return x.tail(n = x.shape[0] -1)

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
	check_points = prediction_deadline - timedelta(n) # 往prediction_deadline前n天內
	x = x[ (x.date >= check_points )]
	return x

#==============================================================================
# def
#==============================================================================
def make(T):
	"""
	T = 0
	folder = 'trainW-0'
	"""
	input_col = ['msno','date']
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
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(user_logs)
	df = pd.merge(train,user_logs, on = 'msno', how = 'left')
	del user_logs
	gc.collect()
	df.sort_values(by = ['msno','date'], inplace = True) # have to do this line for next line
	df['date_diff'] = [i.days for i in (df.date - df['date'].shift(1))]
	print ('shape of df:', df.shape)

	df = df.groupby('msno').apply(drop_first_columns) # 每個user第一欄不用
	df.reset_index(drop = True, inplace = True)

	# ##################################################
	# # All history
	# ##################################################
	# for speed
	# # core
	# tbl = df[df.date_diff == 1].groupby('msno').date_diff.size().to_frame() # date_diff == 1: mean in a row
	# tbl.columns = ['listen_music_in_a_row_count']
	# tbl['listen_music_in_a_row_ratio'] = tbl.listen_music_in_a_row_count / df.groupby('msno').date_diff.apply(len)
	# tbl.reset_index(inplace = True)
	# #==============================================================================
	# print('reduce memory')
	# #==============================================================================
	# utils.reduce_memory(tbl)
	# # write
	# tbl.to_csv('../feature/{}/listen_music_in_a_row_count.csv'.format(folder), index = False)
	# del tbl
	# gc.collect()
	##################################################
	# n = 7
	##################################################
	df_ = df.groupby('msno').apply(within_n_days, T, n = 7).reset_index(drop = True)
	#core
	tbl = df_[df_.date_diff == 1].groupby('msno').date_diff.size().to_frame()
	tbl.columns = ['listen_music_in_a_row_count_during_t_7']
	tbl['listen_music_in_a_row_ratio_during_t_7'] = tbl.listen_music_in_a_row_count_during_t_7 / df_.groupby('msno').date_diff.apply(len)
	tbl.reset_index(inplace = True)
	del df_
	gc.collect()
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/listen_music_in_a_row_count_during_t_7.csv'.format(folder), index = False)
	del tbl
	gc.collect()
	##################################################
	# n = 14
	##################################################
	df_ = df.groupby('msno').apply(within_n_days, T, n = 14).reset_index(drop = True)
	#core
	tbl = df_[df_.date_diff == 1].groupby('msno').date_diff.size().to_frame()
	tbl.columns = ['listen_music_in_a_row_count_during_t_14']
	tbl['listen_music_in_a_row_ratio_during_t_14'] = tbl.listen_music_in_a_row_count_during_t_14 / df_.groupby('msno').date_diff.apply(len)
	tbl.reset_index(inplace = True)
	del df_
	gc.collect()
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/listen_music_in_a_row_count_during_t_14.csv'.format(folder), index = False)
	del tbl
	gc.collect()
	##################################################
	# n = 30
	##################################################
	df_ = df.groupby('msno').apply(within_n_days, T, n = 30).reset_index(drop = True)
	#core
	tbl = df_[df_.date_diff == 1].groupby('msno').date_diff.size().to_frame()
	tbl.columns = ['listen_music_in_a_row_count_during_t_30']
	tbl['listen_music_in_a_row_ratio_during_t_30'] = tbl.listen_music_in_a_row_count_during_t_30 / df_.groupby('msno').date_diff.apply(len)
	tbl.reset_index(inplace = True)
	del df_
	gc.collect()
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/listen_music_in_a_row_count_during_t_30.csv'.format(folder), index = False)
	del tbl
	gc.collect()
	##################################################
	# n = 60
	##################################################
	df_ = df.groupby('msno').apply(within_n_days, T, n = 60).reset_index(drop = True)
	#core
	tbl = df_[df_.date_diff == 1].groupby('msno').date_diff.size().to_frame()
	tbl.columns = ['listen_music_in_a_row_count_during_t_60']
	tbl['listen_music_in_a_row_ratio_during_t_60'] = tbl.listen_music_in_a_row_count_during_t_60 / df_.groupby('msno').date_diff.apply(len)
	tbl.reset_index(inplace = True)
	del df_
	gc.collect()
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/listen_music_in_a_row_count_during_t_60.csv'.format(folder), index = False)
	del tbl
	gc.collect()
	##################################################
	# n = 90
	##################################################
	df_ = df.groupby('msno').apply(within_n_days, T, n = 90).reset_index(drop = True)
	#core
	tbl = df_[df_.date_diff == 1].groupby('msno').date_diff.size().to_frame()
	tbl.columns = ['listen_music_in_a_row_count_during_t_90']
	tbl['listen_music_in_a_row_ratio_during_t_90'] = tbl.listen_music_in_a_row_count_during_t_90 / df_.groupby('msno').date_diff.apply(len)
	tbl.reset_index(inplace = True)
	del df_
	gc.collect()
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(tbl)
	# write
	tbl.to_csv('../feature/{}/listen_music_in_a_row_count_during_t_90.csv'.format(folder), index = False)
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












