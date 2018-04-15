#!/usr/bin/python3
# -*-coding:utf-8

'''
Created on Fri Dec 1 01:08:45 2017

@author: Ray
'''

from tqdm import tqdm
import numpy as np
import os
from glob import glob
import pandas as pd
import gc


def to_multiple_csv(df, path, split_size = 3):
	# Split large dataframe 
	"""
	path = '../output/create_a_dir'

	wirte '../output/create_a_dir/0.csv'
	      '../output/create_a_dir/1.csv'
	      '../output/create_a_dir/2.csv'
	"""

	if not os.path.isdir(path):
		os.makedirs(path)

	t = 0
	for small_dataframe in tqdm(np.array_split(df, split_size)):
	    # np.array_split: return a list of DataFrame
	    # Reference: https://stackoverflow.com/questions/17315737/split-a-large-pandas-dataframe
	    small_dataframe.to_csv(path+'/{}.csv'.format(t), index = False)
	    t+=1

def read_multiple_csv(path, col = None):

	# glob(path+'/*'): return a list, which consist of each files in path

	if col is None:
	    df = pd.concat([pd.read_csv(f) for f in tqdm(sorted(glob(path+'/*')))])
	else:
	    df = pd.concat([pd.read_csv(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
	return df
def reduce_memory(df, ix_start=0):
    df.fillna(-1, inplace=True)
    if df.shape[0] <= 5000:
        df_ = df.sample(1, random_state=71) # just for testing
    else:
        df_ = df.sample(9999, random_state=71) 
    # using sample 取代int_cols = list(df.select_dtypes(include=['int']).columns[ix_start:])
    # 雖然可能會抽到不好的, 但速度是王道
    ## int
    col_int8 = []
    col_int16 = []
    col_int32 = []
    for c in tqdm(df.columns[ix_start:], miniters=20):
        if df[c].dtype =='O':
            continue
        elif df[c].dtype == 'datetime64[ns]':
            continue
        elif (df_[c] == df_[c].astype(np.int8)).all():
            col_int8.append(c)
        elif (df_[c] == df_[c].astype(np.int16)).all():
            col_int16.append(c)
        elif (df_[c] == df_[c].astype(np.int32)).all():
            col_int32.append(c)
    
    df[col_int8]  = df[col_int8].astype(np.int8)
    df[col_int16] = df[col_int16].astype(np.int16)
    df[col_int32] = df[col_int32].astype(np.int32)
    
    ## float
    col = [c for c in df.dtypes[df.dtypes==np.float64].index]
    df[col] = df[col].astype(np.float32)

    gc.collect()
    