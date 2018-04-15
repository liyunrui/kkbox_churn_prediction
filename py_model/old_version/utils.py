#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

'''

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import time
import gc
from itertools import chain

def mkdir_p(path):
    if not os.path.isdir(path):
        os.makedirs(path)

    
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




def keep_top_item(model_file_name, n_top_features):
    #loaded_model
    import pickle
    loaded_model = pickle.load(open(model_file_name, "rb"))
    #top_features
    importance_dict = loaded_model.get_booster().get_score(importance_type='weight')
    tuples = [(k, importance_dict[k]) for k in importance_dict]
    top_features = [t[0] for t in sorted(tuples, key=lambda x: x[1],reverse = True)[:n_top_features]]
    print (len(top_features) == n_top_features)
    #output_column
    col = ['msno', 'is_churn']
    # adding top_feature to output_column
    col += top_features
    return col
    
def load_pred_feature(name, model_file_name, n_top_features, keep_all = False):
    
    if keep_all == False:
        #==============================================================================
        print('keep top imp')
        #==============================================================================
        col = keep_top_item(model_file_name = model_file_name, n_top_features = n_top_features)
        if name=='test':
            col.remove('is_churn') # feature中沒有is_churn
        df = read_multiple_csv('../feature/{}/all'.format(name), col)
        #df = read_multiple_csv('../feature/{}/all_sampling_for_developing'.format(name).format(name), col)
    else:
        #path = '../feature/{}/all_sampling_for_developing'.format(name)
        df = read_multiple_csv('../feature/{}/all'.format(name)) 
    
    print('{}.shape:{}\n'.format(name, df.shape))
    
    return df



#==============================================================================
# main
#==============================================================================
if __name__ == "__main__":
    pass

