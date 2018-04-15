#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

concating features

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

# setting
DATE_churn = ['1215_1', '1215_2', '1215_3','1215_4']

OUTF = "../output/sub/final/bagging.csv.gz"

print("""#==== print param ======""")
print('OUTF:', OUTF)
print('DATE_churn:', DATE_churn)

utils.mkdir_p('../output/sub/final')
#==============================================================================
# load
#==============================================================================
sub_is_churn = pd.concat([pd.read_csv('../output/sub/{}/sub_test.csv'.format(d)) for d in DATE_churn])
sub_is_churn = sub_is_churn.groupby('msno').is_churn.mean().reset_index()
#==============================================================================
# output
#==============================================================================
print('writing...')
sub_is_churn[['msno', 'is_churn']].to_csv(OUTF, index=False, compression='gzip')





