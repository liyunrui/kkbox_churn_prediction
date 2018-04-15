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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from numpy import sort
from sklearn.feature_selection import SelectFromModel # for feature selection


seed = 72
np.random.seed(seed)

##########################################
#load dataset
##########################################

# load dataset
train_0 = utils.load_pred_feature('trainW-0', keep_all = True)
train_1 = utils.load_pred_feature('trainW-1', keep_all = True)
train_2 = utils.load_pred_feature('trainW-2', keep_all = True)
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
Y_train = train['is_churn'] 
X_train = train.drop('is_churn', axis=1)
del train

print ('prepartion of training set is done')


#==============================================================================
# SPLIT : Hold-out validation: only valid 1 times (for saving time)
#==============================================================================
'''
Hold-out validation vs K-fold cross-validation

Hold-out:
	- Pros:
		Only needs to be run once so has lower computational costs. It saves time.
	= Cons:
		lower generalization, it may could lead to overfitting on test set.
K-fold:
	- Pros:
		Robust to noise becuase it uses the entire training set. That is, better generalization
	- Cons:
		The model needs to be trained K times. In other words, it's  time-consuming.

In summarry,
K-fold is super expensive but better at gernerlization.
On the other hand, hold-out is sort of an "approximation" to what k-fold(K = 1) does with more cheaper computation, but worse at gernerlization.

Reference:
1. https://www.kdnuggets.com/2017/08/dataiku-predictive-model-holdout-cross-validation.html
2. https://stats.stackexchange.com/questions/104713/hold-out-validation-vs-cross-validation
'''
def split_build_valid(valid_size = 0.05):
    #--------------------------
    # train/val split by user
    #--------------------------
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

train_user = X_train[['msno']].drop_duplicates()
x_train, y_train, x_val, y_val = split_build_valid(valid_size = 0.05)
#==============================================================================
# Feature selection
#==============================================================================


'''
Core: essentially allowing us to test each subset of features by importance,
starting with all features and ending with a subset with the most important feature.

Operation procedures:
1. using the whole entire dataset to train a base model
2. Also, to get sorted feature importance: From the smallest to the largest, to feed into selection
3. Create a SelectFromModel instance
4. Using method of instance transfrom to reduce X to the selected features.
5. Creat a new classifier trained from the selected subset of features
6. evaluate the classifier

Notice: According to the following plotted figure, 
先初步的找出適合range of n_top_features, 可以之後再做第二次feature selection來找到更趨近最佳的n_top_features
'''
##########################################
# base_model for getting feature importance as thereshold
##########################################
s = time.time()
base_model = XGBClassifier(objective = 'binary:logistic',
                      )
base_model.fit(x_train, y_train, 
          eval_metric ='logloss' ,eval_set = [(x_val, y_val)],
          early_stopping_rounds = 10) 
e = time.time()
print ('Base_model tooks',e-s, 'secs') # Base_model tooks 4520.558794975281 secs

# Fit model using each importance as a threshold
thresholds = sort(base_model.feature_importances_).tolist()# thresh becomes larger

s = time.time()
#acc_num_feature_plot = []
log_loss_num_feature_plot = []
for i,thresh in enumerate(thresholds[::1]):# a[start:end:step]: start through not past end, by step
    # to find suitable thresh
    thresh_t1 = thresholds[i-1]
    if thresh == 0:
        continue
    if thresh_t1 == thresh:
        continue
    print('thresh',thresh)

    # Hold-out validation
    x_train, y_train, x_val, y_val = split_build_valid(valid_size = 0.05)
    # selection : SelectFromModel instance
    selection = SelectFromModel(base_model, threshold = thresh, prefit = True)
    '''
    prefit: Whether a pre-fit model is expected to be passed into the constructor directly or not. 
    threshold: The threshold value to use for feature selection. Features whose importance is greater or equal are kept 
    while the others are discarded. 
 
    '''
    # select features using threshold
    select_X_train = selection.transform(x_train) # Reduce X to the selected features.
    select_X_test = selection.transform(x_val) 
    # train model
    selection_model = XGBClassifier(seed = 72) # fixed seed
    selection_model.fit(select_X_train, y_train)
    # eval model
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    #accuracy = accuracy_score(y_val, predictions)
    logloss = log_loss(y_val, predictions, normalize = False) # return the sum of the per-sample losses, same as xgboost
    #acc_num_feature_plot.append((select_X_train.shape[1],accuracy))
    log_loss_num_feature_plot.append((select_X_train.shape[1],logloss))

    '''
    Honestely, It may be a more robust strategy on a larger dataset using cross validation as the model evaluation scheme.
    However, it's time-consuming. And later, we will tune feature selection again on smaller range of n_top_features.
    Therefore, here we use hold-out cross-validation for quickly reducing # features for following hyperparameter-tuning.
    '''
    print("Thresh={}, n= {}, logloss: {}".format(thresh, select_X_train.shape[1], logloss))

e = time.time()
print ('Feature selection tooks',e-s, 'secs')
####################################
# plot
####################################
import matplotlib.pyplot as plt
x = [p[0] for p in log_loss_num_feature_plot]
y = [p[1] for p in log_loss_num_feature_plot]
plt.plot(x,y)
plt.title('Feature selection via feature importance')
plt.ylabel('acc')
plt.xlabel('top importance features')
plt.show()
# why we need feature selection
# 1. to filter the irrelevant feature improving accuracy
# 2. reducing # features increses speed of computation


####################################
# save base_model for subsequently determing n_top_features
####################################
file_path = '../output/model/xgb_feature_tuning_seed_{}.model'.format(seed)

import pickle # for saving
# save model to file
pickle.dump(base_model, open(file_path, "wb"))


