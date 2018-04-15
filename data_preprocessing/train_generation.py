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
# Load transaction 
##################################################
col = ['msno','transaction_date','membership_expire_date','is_cancel']
transactions = utils.read_multiple_csv('../input/preprocessed_data/transactions', col)

#transactions = transactions.head(n = 5000)
##################################################
# Convert string to datetime format
##################################################
transactions['membership_expire_date']  = transactions.membership_expire_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
transactions['transaction_date']  = transactions.transaction_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

#################################################
#Load train set user and test set user
#################################################
train = pd.read_csv('../input/train_v2.csv') #只用這個檔案, 然後做data augmentation
#test = pd.read_csv('../input/sample_submission_v2.csv')

col = ['msno','transaction_date','membership_expire_date', 'is_churn'] 

train = pd.merge(train, transactions, on = 'msno', how = 'left')[col]

train = train.dropna()

gc.collect()

print ('After Loading ...')

def filter_with_window_size(x, w = 0):
    # w: window_size(short_window_back)
    if w == 0:
        # for data cleaning: remove weird label
        df = x[(x.membership_expire_date >= datetime.strptime('2017-02-01', '%Y-%m-%d')  )
                     &(x.membership_expire_date <= datetime.strptime('2017-02-28', '%Y-%m-%d')  )
                     ] # 找出這個人在17年二月份中的有到期的交易紀錄
        if df.empty:
            # 如果這個交易紀錄是空的,把它濾掉
            x['key'] = 1 # key = 1, 濾掉
        else:
            x['key'] = 0
            x['w'] = w # # 代表這群人將會被預測在17年三月是否流失(true label is provided by train_v2)
        return x
    elif w == 1:
        df = x[(x.membership_expire_date >= datetime.strptime('2017-01-01', '%Y-%m-%d')  )
                     &(x.membership_expire_date <= datetime.strptime('2017-01-31', '%Y-%m-%d')  )
                     ] # 找出這個人在17年一月份中的有到期的交易紀錄
        if df.empty:
            # 如果這個交易紀錄是空的,把它濾掉
            x['key'] = 1 # key = 1, 濾掉
        else:
            x['key'] = 0
        return x
    elif w == 2:
        df = x[(x.membership_expire_date >= datetime.strptime('2016-12-01', '%Y-%m-%d')  )
                     &(x.membership_expire_date <= datetime.strptime('2016-12-31', '%Y-%m-%d')  )
                     ] # 找出這個人在16年十二月份中的有到期的交易紀錄
        if df.empty:
            # 如果這個交易紀錄是空的,把它濾掉
            x['key'] = 1 # key = 1, 濾掉
        else:
            x['key'] = 0
        return x
    
def label_generation(x, w):
    
    if w == 1:
        # 1. 先判斷哪一筆訂單的到期日是落17年在1月中
        df = x[(x.membership_expire_date >= datetime.strptime('2017-01-01', '%Y-%m-%d')  )
                         &(x.membership_expire_date <= datetime.strptime('2017-01-31', '%Y-%m-%d')  )
                         ].tail(n = 1) #如果df有兩筆,取最後一筆
        # 2.find check point
        df['check_point'] = df.membership_expire_date + timedelta(30)
        # 3.找出哪一筆訂單開始在scope of prediction(2月)的order會被拿來當作判斷是否churn的依據
        future = x[(x.transaction_date >= datetime.strptime('2017-02-01', '%Y-%m-%d'))
                   &(x.transaction_date <= datetime.strptime('2017-02-28', '%Y-%m-%d'))
                  ].tail(n = 1)  
        # 4. 判斷is_churn
        if future.empty:
            pass
        else:
            if future.transaction_date.iloc[0] < df.check_point.iloc[0] and future.is_cancel.iloc[0] == 0:
                x['is_churn'] = 0
            else:
                x['is_churn'] = 1
        return x
    elif w == 2:
        # 1. 先判斷哪一筆訂單的到期日是落在16年12月中
        df = x[(x.membership_expire_date >= datetime.strptime('2016-12-01', '%Y-%m-%d')  )
                     &(x.membership_expire_date <= datetime.strptime('2016-12-31', '%Y-%m-%d')  )
                     ].tail(n = 1) 
        # 2.find check point
        df['check_point'] = df.membership_expire_date + timedelta(30)
        # 3.找出哪一筆訂單開始在scope of prediction(17年1月)的order會被拿來當作判斷是否churn的依據
        future = x[(x.transaction_date >= datetime.strptime('2017-01-01', '%Y-%m-%d'))
                   &(x.transaction_date <= datetime.strptime('2017-01-31', '%Y-%m-%d'))
                  ].tail(n = 1)  
        # 4. 判斷is_churn
        if future.empty:
            pass
        else:
            if future.transaction_date.iloc[0] < df.check_point.iloc[0] and future.is_cancel.iloc[0] == 0:
                x['is_churn'] = 0
            else:
                x['is_churn'] = 1
        return x

def make(W):
    output_col = ['msno','is_churn']
    generate_label_col = ['msno','transaction_date','membership_expire_date','is_cancel']
    
    if W == 0:
        global train
        df = train.groupby('msno').apply(filter_with_window_size, w = W)
        del train
        df = df[df.key != 1][output_col].reset_index(drop=True).drop_duplicates('msno', keep = 'last')
        gc.collect()
        df['w'] = W      
        df.to_csv('../input/preprocessed_data/trainW-{0}.csv'.format(W), index = False)
        print ('shape of trainW-{0}'.format(W), df.shape)
        del df
        print ('{0} done'.format(W))
        gc.collect()
    elif W == 1:
        df = transactions.groupby('msno').apply(filter_with_window_size, w = W)
        df = df[df.key != 1][generate_label_col].reset_index(drop=True).groupby('msno').apply(label_generation, w = W)[output_col].dropna()
        df = df.drop_duplicates('msno', keep = 'last')
        df['w'] = W # w = 1代表這群人將會被預測在17年二月是否流失(true label is generated by below function)        
        print ('shape of trainW-{0}'.format(T), df.shape)
        gc.collect()
        # write
        df.to_csv('../input/preprocessed_data/trainW-{0}.csv'.format(W), index = False)
        # path = '../input/preprocessed_data/trainW-{0}'.format(T)
        # utils.to_multiple_csv(
        #     df,
        #     path, 
        #     split_size = 4)

        del df
        print ('{0} done'.format(W))
        gc.collect()    
    elif W == 2:
        df = transactions.groupby('msno').apply(filter_with_window_size, w = W)
        df = df[df.key != 1][generate_label_col].reset_index(drop=True).groupby('msno').apply(label_generation, w = W)[output_col].dropna()
        df = df.drop_duplicates('msno', keep = 'last')
        df['w'] = W # w = 1代表這群人將會被預測在17年二月是否流失(true label is generated by below function)       
        print ('shape of trainW-{0}'.format(W), df.shape)
        gc.collect()

        # write
        df.to_csv('../input/preprocessed_data/trainW-{0}.csv'.format(W), index = False)
        # path = '../input/preprocessed_data/trainW-{0}'.format(T)
        # utils.to_multiple_csv(
        #     df,
        #     path, 
        #     split_size = 4)
        del df
        print ('{0} done'.format(W))
# 優化措施(Memory)
# 1. del沒再用到dataframe, 用gc.collect輔助
# 2. 沒再用到col不用寫出去(output),也不用讀進來(input)
# 3. 把w=1和w=2(資料rows超過1GB,切割儲存)
# 4. 中間過程的欄位drop掉記憶體省的量

# 優化措施(Speed)
# multi processing (多核心運算): 是否要用多核心需要再速度和記一直之間權衡。
# For example, 當你跑一個迴圈要兩次,用平行可以同時跑這回圈,
# 但如果你迴圈中兩遍的記憶體太大,反而單獨跑完一個再跑一個才比較快
##################################################
# Main
##################################################
s = time.time()
make(0) # 0.67 hours
make(1) # 
make(2) # 2.34 hours
# for T in [1,2]:
# 	make(T)
# mp_pool = mp.Pool(4) # going into the pool, python will do multi processing automatically.
# mp_pool.map(make, [0,1,2])

e = time.time()
print (e-s)
