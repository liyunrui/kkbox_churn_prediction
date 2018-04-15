#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

concating features
It took 44 mins
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

#==============================================================================
# def
#==============================================================================
def transactions_feature(df, name):
    '''
    if T ==-1:
        name = 'test'
    else:
        name = 'trainT-'+str(T)
    '''
    #days_since_the_first_subscription
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_first_subscription.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_first_subscription_n5.csv'.format(name)), 
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_first_subscription_n1.csv'.format(name)),
                  on='msno', how='left')
    #days_since_the_last_expiration-cumsum
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_expiration-cumsum.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_expiration-cumsum_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_expiration-cumsum_n1.csv'.format(name)),
                  on='msno', how='left')    
    #days_since_the_last_expiration
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_expiration.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_expiration_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_expiration_n1.csv'.format(name)),
                  on='msno', how='left')    
    #days_since_the_last_expiration_ratio
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_expiration_ratio.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_expiration_ratio_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_expiration_ratio_n1.csv'.format(name)),
                  on='msno', how='left')    
    #days_since_the_last_subscription
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_subscription.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_subscription_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_subscription_n1.csv'.format(name)),
                  on='msno', how='left')    
    #days_since_the_last_subscription_ratio
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_subscription_ratio.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_subscription_ratio_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/days_since_the_last_subscription_ratio_n1.csv'.format(name)),
                  on='msno', how='left')   
    #discount
    df = pd.merge(df, pd.read_csv('../feature/{}/discount.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/discount_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/discount_n1.csv'.format(name)),
                  on='msno', how='left')   
    #do_change_payment_method
    df = pd.merge(df, pd.read_csv('../feature/{}/do_change_payment_method.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/do_change_payment_method_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/do_change_payment_method_n1.csv'.format(name)),
                  on='msno', how='left')   
    #do_extend_payment_days
    df = pd.merge(df, pd.read_csv('../feature/{}/do_extend_payment_days.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/do_extend_payment_days_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/do_extend_payment_days_n1.csv'.format(name)),
                  on='msno', how='left')   
    #do_paid_more
    df = pd.merge(df, pd.read_csv('../feature/{}/do_paid_more.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/do_paid_more_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/do_paid_more_n1.csv'.format(name)),
                  on='msno', how='left')   
    #do_spend_more_money
    df = pd.merge(df, pd.read_csv('../feature/{}/do_spend_more_money.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/do_spend_more_money_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/do_spend_more_money_n1.csv'.format(name)),
                  on='msno', how='left')   
    #early_expiration_days
    df = pd.merge(df, pd.read_csv('../feature/{}/early_expiration_days.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/early_expiration_days_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/early_expiration_days_n1.csv'.format(name)),
                  on='msno', how='left')   
    #is_auto_renew
    df = pd.merge(df, pd.read_csv('../feature/{}/is_auto_renew.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/is_auto_renew_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/is_auto_renew_n1.csv'.format(name)),
                  on='msno', how='left')   
    #is_cancel
    df = pd.merge(df, pd.read_csv('../feature/{}/is_cancel.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/is_cancel_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/is_cancel_n1.csv'.format(name)),
                  on='msno', how='left')   
    #is_discount
    df = pd.merge(df, pd.read_csv('../feature/{}/is_discount.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/is_discount_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/is_discount_n1.csv'.format(name)),
                  on='msno', how='left')   
    #is_subscribe_early
    df = pd.merge(df, pd.read_csv('../feature/{}/is_subscribe_early.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/is_subscribe_early_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/is_subscribe_early_n1.csv'.format(name)),
                  on='msno', how='left')   
    #membership_duration
    df = pd.merge(df, pd.read_csv('../feature/{}/membership_duration.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/membership_duration_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/membership_duration_n1.csv'.format(name)),
                  on='msno', how='left')   
    #over_deadline
    df = pd.merge(df, pd.read_csv('../feature/{}/over_deadline.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/over_deadline_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/over_deadline_n1.csv'.format(name)),
                  on='msno', how='left')   
    #regular_membership
    df = pd.merge(df, pd.read_csv('../feature/{}/regular_membership.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/regular_membership_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/regular_membership_n1.csv'.format(name)),
                  on='msno', how='left')   
    #amt_per_day
    df = pd.merge(df, pd.read_csv('../feature/{}/amt_per_day.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/amt_per_day_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/amt_per_day_n1.csv'.format(name)),
                  on='msno', how='left')   
    #cp_value
    df = pd.merge(df, pd.read_csv('../feature/{}/cp_value.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/cp_value_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/cp_value_n1.csv'.format(name)),
                  on='msno', how='left')      
    gc.collect() 
    return df
def members_feature(df, name):
    '''
    if T ==-1:
        name = 'test'
    else:
        name = 'trainT-'+str(T)
    '''
    #membership_stat
    df = pd.merge(df, pd.read_csv('../feature/{}/membership_stat.csv'.format(name)),
                  on='msno', how='left')
    gc.collect()
    return df

def user_logs_feature(df, name):   
    #num_25
    df = pd.merge(df, pd.read_csv('../feature/{}/num_25.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_25_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_25_n1.csv'.format(name)),
                  on='msno', how='left')       
    #num_50
    df = pd.merge(df, pd.read_csv('../feature/{}/num_50.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_50_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_50_n1.csv'.format(name)),
                  on='msno', how='left')   
    #num_75
    df = pd.merge(df, pd.read_csv('../feature/{}/num_75.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_75_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_75_n1.csv'.format(name)),
                  on='msno', how='left')       
    #num_985
    df = pd.merge(df, pd.read_csv('../feature/{}/num_985.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_985_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_985_n1.csv'.format(name)),
                  on='msno', how='left')   
    #num_100
    df = pd.merge(df, pd.read_csv('../feature/{}/num_100.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_100_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_100_n1.csv'.format(name)),
                  on='msno', how='left')    
    #num_unq
    df = pd.merge(df, pd.read_csv('../feature/{}/num_unq.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_unq_n5.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_unq_n1.csv'.format(name)),
                  on='msno', how='left') 
    #completed_songs_ratio
    df = pd.merge(df, pd.read_csv('../feature/{}/completed_songs_ratio.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/completed_songs_ratio_during_t_7.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/completed_songs_ratio_during_t_14.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/completed_songs_ratio_during_t_30.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/completed_songs_ratio_during_t_60.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/completed_songs_ratio_during_t_90.csv'.format(name)),
                  on='msno', how='left') 
    #date_diff
    df = pd.merge(df, pd.read_csv('../feature/{}/date_diff.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/date_diff_during_t_7.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/date_diff_during_t_14.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/date_diff_during_t_30.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/date_diff_during_t_60.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/date_diff_during_t_90.csv'.format(name)),
                  on='msno', how='left') 
    #listen_music_in_a_row_count
    # df = pd.merge(df, pd.read_csv('../feature/{}/listen_music_in_a_row_count.csv'.format(name)),
    #               on='msno', how='left') # no whole history
    df = pd.merge(df, pd.read_csv('../feature/{}/listen_music_in_a_row_count_during_t_7.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/listen_music_in_a_row_count_during_t_14.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/listen_music_in_a_row_count_during_t_30.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/listen_music_in_a_row_count_during_t_60.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/listen_music_in_a_row_count_during_t_90.csv'.format(name)),
                  on='msno', how='left') 
    #num_100_ratio
    df = pd.merge(df, pd.read_csv('../feature/{}/num_100_ratio.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_100_ratio_during_t_7.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_100_ratio_during_t_14.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/num_100_ratio_during_t_30.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_100_ratio_during_t_60.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/num_100_ratio_during_t_90.csv'.format(name)),
                  on='msno', how='left') 
    #num_25_ratio
    df = pd.merge(df, pd.read_csv('../feature/{}/num_25_ratio.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_25_ratio_during_t_7.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_25_ratio_during_t_14.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/num_25_ratio_during_t_30.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_25_ratio_during_t_60.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/num_25_ratio_during_t_90.csv'.format(name)),
                  on='msno', how='left') 
    #num_log_in
    df = pd.merge(df, pd.read_csv('../feature/{}/num_log_in.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_log_in_during_t_7.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_log_in_during_t_14.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/num_log_in_during_t_30.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_log_in_during_t_60.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/num_log_in_during_t_90.csv'.format(name)),
                  on='msno', how='left') 
    #num_repeated_songs
    df = pd.merge(df, pd.read_csv('../feature/{}/num_repeated_songs.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_repeated_songs_during_t_7.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_repeated_songs_during_t_14.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/num_repeated_songs_during_t_30.csv'.format(name)),
                  on='msno', how='left')
    df = pd.merge(df, pd.read_csv('../feature/{}/num_repeated_songs_during_t_60.csv'.format(name)),
                  on='msno', how='left') 
    df = pd.merge(df, pd.read_csv('../feature/{}/num_repeated_songs_during_t_90.csv'.format(name)),
                  on='msno', how='left') 
    gc.collect()
    return df
def concat_pred_features(T):
    if T == -1:
        name = 'test'
        train = pd.read_csv('../input/sample_submission_v2.csv') # 此train代表的是test的user
    else:
        #==============================================================================
        print('load label')
        #==============================================================================        
        name = 'trainW-'+str(T)
        train = pd.read_csv('../input/preprocessed_data/trainW-{0}.csv'.format(T))[['msno','is_churn']] 

    #train = train.head( n = 500)
    #==============================================================================
    print('transactions feature')
    #==============================================================================
    df = transactions_feature(train, name)
    
    print('{}.shape:{}\n'.format(name, df.shape))

    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(df)
    ix_end = df.shape[1]
         
    #==============================================================================
    print('members feature')
    #==============================================================================
    df = members_feature(df, name)
    print('{}.shape:{}\n'.format(name, df.shape))

    #==============================================================================
    print('user_logs feature')
    #==============================================================================
    df = user_logs_feature(df, name)
    df.replace(np.inf, 0, inplace = True) # It may destroy feature but forget it. just noise
    print('{}.shape:{}\n'.format(name, df.shape))

    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(df, ix_end) 
    ix_end = df.shape[1]

    #==============================================================================
    print('feature engineering')
    #==============================================================================
    # delta:反應這個值隨者時間的變化量(future - history)
    # delta1: time difference = 1
    # delta2: time difference = 2
    # bynum: start from the num
    ####
    #num_100_ratio-mean
    ####
    #delta1
    df['num_100_ratio_delta1_by7'] = df['num_100_ratio_during_t_7-mean'] - df['num_100_ratio_during_t_14-mean'] 
    df['num_100_ratio_delta1_by14'] = df['num_100_ratio_during_t_14-mean'] - df['num_100_ratio_during_t_30-mean']
    df['num_100_ratio_delta1_by30'] = df['num_100_ratio_during_t_30-mean'] - df['num_100_ratio_during_t_60-mean']
    df['num_100_ratio_delta1_by60'] = df['num_100_ratio_during_t_60-mean'] - df['num_100_ratio_during_t_90-mean']
    df['num_100_ratio_delta1_by90'] = df['num_100_ratio_during_t_90-mean'] - df['num_100_ratio-mean']
    #delta2
    df['num_100_ratio_delta2_by7'] = df['num_100_ratio_during_t_7-mean'] - df['num_100_ratio_during_t_30-mean']
    df['num_100_ratio_delta2_by14'] = df['num_100_ratio_during_t_14-mean'] - df['num_100_ratio_during_t_60-mean']
    df['num_100_ratio_delta2_by30'] = df['num_100_ratio_during_t_30-mean'] - df['num_100_ratio_during_t_90-mean']
    df['num_100_ratio_delta2_by60'] = df['num_100_ratio_during_t_60-mean'] - df['num_100_ratio-mean']
    ####
    #num_25_ratio-mean
    ####
    #delta1
    df['num_25_ratio_delta1_by7'] = df['num_25_ratio_during_t_7-mean'] - df['num_25_ratio_during_t_14-mean']
    df['num_25_ratio_delta1_by14'] = df['num_25_ratio_during_t_14-mean'] - df['num_25_ratio_during_t_30-mean']
    df['num_25_ratio_delta1_by30'] = df['num_25_ratio_during_t_30-mean'] - df['num_25_ratio_during_t_60-mean']
    df['num_25_ratio_delta1_by60'] = df['num_25_ratio_during_t_60-mean'] - df['num_25_ratio_during_t_90-mean']
    df['num_25_ratio_delta1_by90'] = df['num_25_ratio_during_t_90-mean'] - df['num_25_ratio-mean']
    #delta2
    df['num_25_ratio_delta2_by7'] = df['num_25_ratio_during_t_7-mean'] - df['num_25_ratio_during_t_30-mean']
    df['num_25_ratio_delta2_by14'] = df['num_25_ratio_during_t_14-mean'] - df['num_25_ratio_during_t_60-mean']
    df['num_25_ratio_delta2_by30'] = df['num_25_ratio_during_t_30-mean'] - df['num_25_ratio_during_t_90-mean']
    df['num_25_ratio_delta2_by60'] = df['num_25_ratio_during_t_60-mean'] - df['num_25_ratio-mean']
    ####
    #num_repeated_songs-mean
    ####
    #delta1
    df['num_repeated_songs_delta1_by7'] = df['num_repeated_songs_during_t_7-mean'] - df['num_repeated_songs_during_t_14-mean']
    df['num_repeated_songs_delta1_by14'] = df['num_repeated_songs_during_t_14-mean'] - df['num_repeated_songs_during_t_30-mean']
    df['num_repeated_songs_delta1_by30'] = df['num_repeated_songs_during_t_30-mean'] - df['num_repeated_songs_during_t_60-mean']
    df['num_repeated_songs_delta1_by60'] = df['num_repeated_songs_during_t_60-mean'] - df['num_repeated_songs_during_t_90-mean']
    df['num_repeated_songs_delta1_by90'] = df['num_repeated_songs_during_t_90-mean'] - df['num_repeated_songs-mean']
    #delta2
    df['num_repeated_songs_delta2_by7'] = df['num_repeated_songs_during_t_7-mean'] - df['num_repeated_songs_during_t_30-mean']
    df['num_repeated_songs_delta2_by14'] = df['num_repeated_songs_during_t_14-mean'] - df['num_repeated_songs_during_t_60-mean']
    df['num_repeated_songs_delta2_by30'] = df['num_repeated_songs_during_t_30-mean'] - df['num_repeated_songs_during_t_90-mean']
    df['num_repeated_songs_delta2_by60'] = df['num_repeated_songs_during_t_60-mean'] - df['num_repeated_songs-mean']
    ####
    #completed_songs_ratio
    ####
    #delta1
    df['completed_songs_ratio_delta1_by7'] = df['completed_songs_ratio_during_t_7-mean'] - df['completed_songs_ratio_during_t_14-mean']
    df['completed_songs_ratio_delta1_by14'] = df['completed_songs_ratio_during_t_14-mean'] - df['completed_songs_ratio_during_t_30-mean']
    df['completed_songs_ratio_delta1_by30'] = df['completed_songs_ratio_during_t_30-mean'] - df['completed_songs_ratio_during_t_60-mean']
    df['completed_songs_ratio_delta1_by60'] = df['completed_songs_ratio_during_t_60-mean'] - df['completed_songs_ratio_during_t_90-mean']
    df['completed_songs_ratio_delta1_by90'] = df['completed_songs_ratio_during_t_90-mean'] - df['completed_songs_ratio-mean']
    #delta2
    df['completed_songs_ratio_delta2_by7'] = df['completed_songs_ratio_during_t_7-mean'] - df['completed_songs_ratio_during_t_30-mean']
    df['completed_songs_ratio_delta2_by14'] = df['completed_songs_ratio_during_t_14-mean'] - df['completed_songs_ratio_during_t_60-mean']
    df['completed_songs_ratio_delta2_by30'] = df['completed_songs_ratio_during_t_30-mean'] - df['completed_songs_ratio_during_t_90-mean']
    df['completed_songs_ratio_delta2_by60'] = df['completed_songs_ratio_during_t_60-mean'] - df['completed_songs_ratio-mean']
    ####
    #listen_music_in_a_row_count
    ####
    #delta1
    df['listen_music_in_a_row_count_delta1_by7'] = df['listen_music_in_a_row_count_during_t_7'] - df['listen_music_in_a_row_count_during_t_14']
    df['listen_music_in_a_row_count_delta1_by14'] = df['listen_music_in_a_row_count_during_t_14'] - df['listen_music_in_a_row_count_during_t_30']
    df['listen_music_in_a_row_count_delta1_by30'] = df['listen_music_in_a_row_count_during_t_30'] - df['listen_music_in_a_row_count_during_t_60']
    df['listen_music_in_a_row_count_delta1_by60'] = df['listen_music_in_a_row_count_during_t_60'] - df['listen_music_in_a_row_count_during_t_90']
    #delta2
    df['listen_music_in_a_row_count_delta2_by7'] = df['listen_music_in_a_row_count_during_t_7'] - df['listen_music_in_a_row_count_during_t_30']
    df['listen_music_in_a_row_count_delta2_by14'] = df['listen_music_in_a_row_count_during_t_14'] - df['listen_music_in_a_row_count_during_t_60']
    df['listen_music_in_a_row_count_delta2_by30'] = df['listen_music_in_a_row_count_during_t_30'] - df['listen_music_in_a_row_count_during_t_90']
    
    ####
    #listen_music_in_a_row_ratio
    ####
    #delta1
    df['listen_music_in_a_row_ratio_delta1_by7'] = df['listen_music_in_a_row_ratio_during_t_7'] - df['listen_music_in_a_row_ratio_during_t_14']
    df['listen_music_in_a_row_ratio_delta1_by14'] = df['listen_music_in_a_row_ratio_during_t_14'] - df['listen_music_in_a_row_ratio_during_t_30']
    df['listen_music_in_a_row_ratio_delta1_by30'] = df['listen_music_in_a_row_ratio_during_t_30'] - df['listen_music_in_a_row_ratio_during_t_60']
    df['listen_music_in_a_row_ratio_delta1_by60'] = df['listen_music_in_a_row_ratio_during_t_60'] - df['listen_music_in_a_row_ratio_during_t_90']
    #delta2
    df['listen_music_in_a_row_ratio_delta2_by7'] = df['listen_music_in_a_row_ratio_during_t_7'] - df['listen_music_in_a_row_ratio_during_t_30']
    df['listen_music_in_a_row_ratio_delta2_by14'] = df['listen_music_in_a_row_ratio_during_t_14'] - df['listen_music_in_a_row_ratio_during_t_60']
    df['listen_music_in_a_row_ratio_delta2_by30'] = df['listen_music_in_a_row_ratio_during_t_30'] - df['listen_music_in_a_row_ratio_during_t_90']
    ####
    #date_diff-mean
    ####
    #delta1
    df['date_diff_delta1_by7'] = df['date_diff_during_t_7-mean'] - df['date_diff_during_t_14-mean']
    df['date_diff_delta1_by14'] = df['date_diff_during_t_14-mean'] - df['date_diff_during_t_30-mean']
    df['date_diff_delta1_by30'] = df['date_diff_during_t_30-mean'] - df['date_diff_during_t_60-mean']
    df['date_diff_delta1_by60'] = df['date_diff_during_t_60-mean'] - df['date_diff_during_t_90-mean']
    df['date_diff_delta1_by90'] = df['date_diff_during_t_90-mean'] - df['date_diff-mean']
    #delta2
    df['date_diff_delta2_by7'] = df['date_diff_during_t_7-mean'] - df['completed_songs_ratio_during_t_30-mean']
    df['date_diff_delta2_by14'] = df['date_diff_during_t_14-mean'] - df['date_diff_during_t_60-mean']
    df['date_diff_delta2_by30'] = df['date_diff_during_t_30-mean'] - df['date_diff_during_t_90-mean']
    df['date_diff_delta2_by60'] = df['date_diff_during_t_60-mean'] - df['date_diff-mean']
    ####
    #num_log_in
    ####
    #delta1
    df['num_log_in_delta1_by7'] = df['num_log_in_during_t_7'] - df['num_log_in_during_t_14']
    df['num_log_in_delta1_by14'] = df['num_log_in_during_t_14'] - df['num_log_in_during_t_30']
    df['num_log_in_delta1_by30'] = df['num_log_in_during_t_30'] - df['num_log_in_during_t_60']
    df['num_log_in_delta1_by60'] = df['num_log_in_during_t_60'] - df['num_log_in_during_t_90']
    df['num_log_in_delta1_by90'] = df['num_log_in_during_t_90'] - df['num_log_in']
    #delta2
    df['num_log_in_delta2_by7'] = df['num_log_in_during_t_7'] - df['num_log_in_during_t_30']
    df['num_log_in_delta2_by14'] = df['num_log_in_during_t_14'] - df['num_log_in_during_t_60']
    df['num_log_in_delta2_by30'] = df['num_log_in_during_t_30'] - df['num_log_in_during_t_90']
    df['num_log_in_delta2_by60'] = df['num_log_in_during_t_60'] - df['num_log_in']
    ####
    #log_in_ratio
    ####
    #delta1
    df['log_in_ratio_delta1_by7'] = df['log_in_ratio_during_t_7'] - df['log_in_ratio_during_t_14']
    df['log_in_ratio_delta1_by14'] = df['log_in_ratio_during_t_14'] - df['log_in_ratio_during_t_30']
    df['log_in_ratio_delta1_by30'] = df['log_in_ratio_during_t_30'] - df['log_in_ratio_during_t_60']
    df['log_in_ratio_delta1_by60'] = df['log_in_ratio_during_t_60'] - df['log_in_ratio_during_t_90']
    df['log_in_ratio_delta1_by90'] = df['log_in_ratio_during_t_90'] - df['log_in_ratio']
    #delta2
    df['log_in_ratio_delta2_by7'] = df['log_in_ratio_during_t_7'] - df['log_in_ratio_during_t_30']
    df['log_in_ratio_delta2_by14'] = df['log_in_ratio_during_t_14'] - df['log_in_ratio_during_t_60']
    df['log_in_ratio_delta2_by30'] = df['log_in_ratio_during_t_30'] - df['log_in_ratio_during_t_90']
    df['log_in_ratio_delta2_by60'] = df['log_in_ratio_during_t_60'] - df['log_in_ratio']

    print('{}.shape:{}\n'.format(name, df.shape))
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(df, ix_end)

    #==============================================================================
    print('output')
    #==============================================================================

    utils.to_multiple_csv(df, '../feature/{}/all'.format(name), 20) # 存一個all_sampling_for_developing   
    #utils.to_multiple_csv(df, '../feature/{}/all_sampling_for_developing'.format(name), 20) # 存一個all_sampling_for_developing

def multi(name):
    concat_pred_features(name)

##################################################
# Main
##################################################
s = time.time()

mp_pool = mp.Pool(4)
mp_pool.map(multi, [0,1,2,-1])

e = time.time()
print (e-s)





