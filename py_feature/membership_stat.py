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


# Note: members.csv IS not inclusive of all users
##################################################
# Load members and
##################################################
demographics = pd.read_csv('../input/preprocessed_data/demographics.csv')

##################################################
# Convert string to datetime format
##################################################
demographics['registration_init_time']  = demographics.registration_init_time.apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))

#==============================================================================
print('reduce memory')
#==============================================================================
utils.reduce_memory(demographics)

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
	else:
		folder = 'trainW-'+str(T)
		train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[['msno']] # we do not need is_churn
	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(train)

	df = pd.merge(train, demographics, on = 'msno',how = 'left')
	if T == 0:
		now_time = datetime.strptime('2017-03-01', '%Y-%m-%d') 
	elif T == 1:
		now_time = datetime.strptime('2017-02-01', '%Y-%m-%d')
	elif T == 2:
		now_time = datetime.strptime('2017-01-01', '%Y-%m-%d')
	else:
		now_time = datetime.strptime('2017-04-01', '%Y-%m-%d')
	df['how_long_has_benn_a_memmbership_of_kkbox_days'] = [(now_time - datetime.utcfromtimestamp(r_i_t.tolist()/1e9)).days if pd.notnull(r_i_t) else -1 for r_i_t in df.registration_init_time.values]
	df['how_long_has_benn_a_memmbership_of_kkbox_years'] = [h_days/360 if h_days != -1 else -1 for h_days in df.how_long_has_benn_a_memmbership_of_kkbox_days.values]
	df.drop('registration_init_time',axis = 1, inplace =True)
	#==============================================================================
	print('one-hot encoding for dummy varaiables') 
	#==============================================================================
	df = pd.get_dummies(df, columns=['city'])
	df = pd.get_dummies(df, columns=['gender'])
	df = pd.get_dummies(df, columns=['registered_via'])
	# the following's value is meaningful, so it do not need one-hot encoding
	# df = pd.get_dummies(df, columns=['city_zone'])
	# df = pd.get_dummies(df, columns=['bd_zone'])
	# df = pd.get_dummies(df, columns=['registered_via_zone'])

	#==============================================================================
	print('reduce memory')
	#==============================================================================
	utils.reduce_memory(df)
	# write
	df.to_csv('../feature/{}/membership_stat.csv'.format(folder), index = False)

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















