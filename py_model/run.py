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

##############
#xgboost model
##############

os.system('python3 -u xgb_churn_1.py')
os.system('python3 -u xgb_churn_2.py')
os.system('python3 -u xgb_churn_3.py')
os.system('python3 -u xgb_churn_4.py')
os.system('python3 -u bagging.py')