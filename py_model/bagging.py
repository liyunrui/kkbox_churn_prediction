#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

'''
import warnings
warnings.filterwarnings("ignore")
from glob import glob
import gc
import os
from tqdm import tqdm
import time
from itertools import chain
import pandas as pd
import numpy as np
from xgboost import plot_importance
from xgboost import XGBClassifier
import utils # made by author for efficiently dealing with data
import pickle # for saving

# setting
DATE_churn = ['0308_71', '0308_72', '0308_73','0308_74']


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

