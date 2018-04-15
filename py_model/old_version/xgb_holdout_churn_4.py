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
import gc
import sys
sys.path.append('/Users/yunruili/xgboost/python-package')
import xgboost as xgb
import utils


# setting
DATE = '1215_4'
LOOP = 2 # larger learning rate, larger number of ensembles
ESR = 40

#seed = np.random.randint(99999)
seed = 72

np.random.seed(seed)

valid_size = 0.05


# XGB param
nround = 10000
#nround = 10

param = {'max_depth':10, 
         'min_child_weight':2,
         'eta':0.02,
         'colsample_bytree':0.4,
         'subsample':0.75,
         'silent':1,
         'nthread':27,
         'eval_metric':'logloss',
         'objective':'binary:logistic',
         'tree_method':'hist'
         }
# reduce eta, can avoid overfitting
# modified: eta: 0.01 --> 0.02; min_child_weight: 1-->2
print("""#==== print param ======""")
print('DATE:', DATE)
print('seed:', seed)

#==============================================================================
# prepare
#==============================================================================
train = pd.concat([utils.load_pred_feature('trainW-0'),
                   utils.load_pred_feature('trainW-1'),
                   utils.load_pred_feature('trainW-2'),
                   ], ignore_index=True)

y_train = train['is_churn']
X_train = train.drop('is_churn', axis=1)
del train
gc.collect()

X_train.fillna(-1, inplace = True)



#==============================================================================
# SPLIT!
print(' train/val splitting by user')
#==============================================================================
train_user = X_train[['msno']].drop_duplicates()

def split_build_valid():
    # train/val split by user
    train_user['is_valid'] = np.random.choice([0,1], size=len(train_user), 
                                              p=[1-valid_size, valid_size])
    # is_valid: 1 if the user is validating user else 0
    valid_n = train_user['is_valid'].sum()
    build_n = (train_user.shape[0] - valid_n)
    
    print('build user:{}, valid user:{}'.format(build_n, valid_n))
    valid_user = train_user[train_user['is_valid']==1].msno
    is_valid = X_train.msno.isin(valid_user)
    
    dbuild = xgb.DMatrix(X_train[~is_valid].drop('msno', axis=1), y_train[~is_valid])
    dvalid = xgb.DMatrix(X_train[is_valid].drop('msno', axis=1), label = y_train[is_valid])
    watchlist = [(dbuild, 'build'),(dvalid, 'valid')]
    
    print('FINAL SHAPE')
    print('dbuild.shape:{}  dvalid.shape:{}\n'.format((dbuild.num_row(), dbuild.num_col()),
                                                      (dvalid.num_row(), dvalid.num_col())))

    return dbuild, dvalid, watchlist

#==============================================================================
print('training')
#==============================================================================
utils.mkdir_p('../output/model/{}/'.format(DATE))
utils.mkdir_p('../output/sub/{}/'.format(DATE))

# hold out
models = [] # for the following prediction
for i in range(LOOP):
    print('LOOP',i)
    dbuild, dvalid, watchlist = split_build_valid()
    
    if i==0:
        col_train = dbuild.feature_names 
        # col_train is built for testing cause u do not use all feature. 
        # we use feature with top importance
    # for watching loss
    model = xgb.train(param, dbuild, nround, watchlist,
                      early_stopping_rounds=ESR, verbose_eval=5)
    ###########
    # # final model(using dbuild + dtrain)
    ###########
    # dtrain = xgb.DMatrix(X_train.drop('msno', axis=1), y_train)
    # model = xgb.train(param, dtrain, nround)

    models.append(model)
    model.save_model('../output/model/{}/xgb_churn_{}.model'.format(DATE, i))
    # VALID
    valid_yhat = model.predict(dvalid) # y_hat is result of prediction
    print('Valid Mean:', np.mean(valid_yhat))
    del dbuild, dvalid, watchlist
    gc.collect()

    

#==============================================================================
print('test')
#==============================================================================
test = utils.load_pred_feature('test').fillna(-1)
sub_test = test[['msno']]

dtest  = xgb.DMatrix(test[col_train])
sub_test['is_churn'] = 0
for model in models:
    sub_test['is_churn'] += model.predict(dtest)
sub_test['is_churn'] /= LOOP
print('Test Mean:', sub_test['is_churn'].mean())

sub_test.to_csv('../output/sub/{}/sub_test.csv'.format(DATE), index = False)


#==============================================================================
