#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Mon Mar 1 2018

@author: Ray

'''

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

###############################
# setting
###############################

# using different random seed to make sure variety of models
seed = 74
np.random.seed(seed) 

DATE = '0308'
utils.mkdir_p('../output/model/{}_{}/'.format(DATE,seed))
utils.mkdir_p('../output/sub/{}_{}/'.format(DATE,seed))

print("""#==== print param ======""")
print('DATE:', DATE)
print('seed:', seed)


##########################################
#load dataset
##########################################

# load dataset
file_name = '../output/model/xgb_feature_tuning_seed_72.model'
train_0 = utils.load_pred_feature('trainW-0', keep_all = False, model_file_name = file_name, n_top_features = 48)
train_1 = utils.load_pred_feature('trainW-1', keep_all = False, model_file_name = file_name, n_top_features = 48)
train_2 = utils.load_pred_feature('trainW-2', keep_all = False, model_file_name = file_name, n_top_features = 48)
# make data augmentation having same label distribution with training set provided by the kkbox
per_churned_in_train_0 = train_0[['is_churn']].describe().ix['mean'][0] 
n_churned = train_1[train_1.is_churn == 0].shape[0] * per_churned_in_train_0
print('per_churned_in_train_0', per_churned_in_train_0)
print('n_churned', int(n_churned))
train_1 = pd.concat([train_1[train_1.is_churn == 0],
                train_1[train_1.is_churn == 1].sample(n = int(n_churned), random_state = seed)
               ], ignore_index=True)
per_churned_in_train_1 = train_1[['is_churn']].describe().ix['mean'][0] 
print('per_churned_in_train_1', per_churned_in_train_1)
train_2 = pd.concat([train_2[train_2.is_churn == 0],
                train_2[train_2.is_churn == 1].sample(n = int(n_churned), random_state = seed)
               ], ignore_index=True)
per_churned_in_train_2 = train_1[['is_churn']].describe().ix['mean'][0] 
print('per_churned_in_train_2', per_churned_in_train_2)
train = pd.concat([train_0, train_1, train_2], ignore_index=True)

# reduce memory in python
del train_0, train_1, train_2, per_churned_in_train_0, per_churned_in_train_1, per_churned_in_train_2, n_churned,
gc.collect()

#==============================================================================
# prepare training data
#==============================================================================
X_train = train.drop('is_churn', axis=1)
Y_train = train['is_churn']
del train
gc.collect()
print ('prepartion of training set is done')

#==============================================================================
print('training')
#==============================================================================

'''
Q: Training with the full dataset after cross-validation?/Re-train on the whole dataset after validating the model?

A: Yes, we re-train the model with the entire dateset because using more date we have is always better.
In case of time-series, including more recent data is always better.
but how can i avoid to over-fitting in the final model. I mean, take early stopping techniques as an example,
we can use evaluation set to early stop to avoid over-fitting, but if we used entire dataset, we cannot do that in the same way.

In practical, people do not use entire dataset to train final model. 
They usually hold a small data set to get CV socre, and trust cv score.

Reference:https://books.google.com.tw/books?id=_plGDwAAQBAJ&pg=PA192&lpg=PA192&dq=re-train+on+the+whole+dataset+after+validating+the+model?&source=bl&ots=8sEFaTdrGJ&sig=NBBFEe9Dspf0AyEK60njR3skPpY&hl=zh-TW&sa=X&ved=0ahUKEwje-KyKn9XZAhVFrJQKHbNEAwkQ6AEIZTAH#v=onepage&q=re-train%20on%20the%20whole%20dataset%20after%20validating%20the%20model%3F&f=false

'''
train_user = X_train[['msno']].drop_duplicates()

def split_build_valid(valid_size = 0.05):
    '''
    Hold-out validation

    parameters:
    ------------------
    valid_size: float

    '''
    # train/val split by user
    train_user['is_valid'] = np.random.choice([0,1], size=len(train_user), 
                                              p=[1-valid_size, valid_size]) # randomly pick someone as validation user
    # is_valid: 1 if the user is validating user else 0
    valid_n = train_user['is_valid'].sum()
    build_n = (train_user.shape[0] - valid_n)
    print('build user:{}, valid user:{}'.format(build_n, valid_n))
    valid_user = train_user[train_user['is_valid']==1].msno
    is_valid = X_train.msno.isin(valid_user)
    # to create the XGBoost matrices that will be used to train the model using XGBoost. 
    x_train = X_train[~is_valid].drop('msno', axis=1)
    y_train = Y_train[~is_valid]
    x_val = X_train[is_valid].drop('msno', axis=1)
    y_val = Y_train[is_valid]
    print('FINAL SHAPE')
    print('x_train.shape:{0}'.format(x_train.shape))
    print('x_val.shape:{0}'.format(x_val.shape))

    return x_train, y_train, x_val, y_val




# xgboost parameters (tuning result from cv)
params_fixed = {
	'objective' : 'binary:logistic',
	'max_depth': 7,
	'min_child_weight': 4,
	'gamma': 0.0,
	'subsample' : 0.95,
	'colsample_bytree': 0.75,
	'reg_lambda': 5,
	'seed': seed
}

#Here we did is lower the learning rate and add more trees.
n_estimators = 10000
learning_rate = 0.01
early_stopping_rounds = 50

# for simple ensemble
LOOP = 2

# Core
models = [] # for the following prediction
for i in range(LOOP):
    print('LOOP',i)
    # hold-out validation
    x_train, y_train, x_val, y_val = split_build_valid(valid_size = 0.05)
    # model training
    model = XGBClassifier(
        **params_fixed,
        n_estimators = n_estimators,
        learning_rate = learning_rate
                             )
    model.fit(x_train, y_train, 
                  eval_metric ='logloss' , eval_set = [(x_val,y_val)],
                  early_stopping_rounds = early_stopping_rounds) 
    # saving
    models.append(model)
    pickle.dump(model, open('../output/model/{}_{}/xgb_churn_{}.model'.format(DATE, seed, i), "wb"))
    # validating
    valid_yhat = model.predict(x_val) # y_hat is result of prediction
    print('Valid Mean:', np.mean(valid_yhat))
    del x_train, y_train, x_val, y_val
    gc.collect()


del X_train, Y_train

#==============================================================================
print('test')
#==============================================================================
#load testing set
test = utils.load_pred_feature('test',keep_all = False, model_file_name = file_name, n_top_features = 48).fillna(-1)
sub_test = test[['msno']]
test.drop('msno', axis = 1, inplace = True) # remove msno for subsequent prediting

#Core
sub_test['is_churn'] = 0 
for model in models:
    sub_test['is_churn'] += model.predict_proba(test)[:,1].clip(min = 0.+1e-15, max = 1-1e-15)
sub_test['is_churn'] /= LOOP # do some simple ensemble: average of prediting result

print('Test Mean:', sub_test['is_churn'].mean())

'''
From observing difference between test and val, we can make sure val and test comes from same distribution.
'''
# saving for submitting
sub_test.to_csv('../output/sub/{}_{}/sub_test.csv'.format(DATE,seed), index = False)

