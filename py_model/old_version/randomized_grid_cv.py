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
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold



train = utils.load_pred_feature('trainW-0')

X = train.drop('is_churn', axis=1)
y = train['is_churn']


# cross_validation strategies
seed = 72
cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)
# grid
params_fixed = {
	         'silent':1,
         'objective':'binary:logistic',
}

params_dist_grid = {
	'max_depth': [5,6,7,8,9,10],
	'min_child_weight': [1,3,5],
	'colsample_bytree': [0.4,0,6,0.8]
}

rs_grid = RandomizedSearchCV(
	estimator = XGBClassifier(**params_fixed, seed = seed),
	param_distributions = params_dist_grid,
	n_iter = 10,
	cv = cv,
	scoring = 'neg_log_loss',
	random_state = seed
	)

rs_grid.fix(X,y)

print (rs_grid.best_estimator_)
print (rs_grid.best_score_)
print (rs_grid.best_params_)