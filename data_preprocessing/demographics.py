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
from collections import Counter

##################################################
# Load members and
##################################################
members = pd.read_csv('../input/members_v3.csv')


##################################################
# Gender
##################################################
def gender(x):
    if x == 'female':
        return 0
    elif x == 'male':
        return 1
    else:
        return 2 # 2:代表missing values
members['gender'] = members.gender.apply(gender)
##################################################
# city
##################################################
members['city'] = members.city.apply(lambda x: int(x) if pd.notnull(x) else 2) # 把City-NaN當作2

##################################################
#registered_via
##################################################
registered_via_dict = {k:v+1 for v,k in enumerate(members['registered_via'].unique().tolist())}
'''
key: original registered_via
{-1: 18,
 1: 15,
 2: 10,
 3: 4,
 4: 6,
 5: 9,
 6: 13,
 7: 2,
 8: 12,
 9: 3,
 10: 17,
 11: 1,
 13: 7,
 14: 14,
 16: 5,
 17: 8,
 18: 16,
 19: 11}
 # 18代表nan
'''
members['registered_via'] = members.registered_via.apply(lambda x: registered_via_dict[x]) 

##################################################
# Birthd Date Cleaning
##################################################

# missing value (12%) and outliers(about 49%)
members['bd'] = members.bd.apply(lambda x: -99999 if float(x)<=1 else x )
members['bd'] = members.bd.apply(lambda x: -99999 if float(x)>=100 else x )
members['bd'] = members.bd.apply(lambda x: int(x) if pd.notnull(x) else -99999 )

tmp_bd_for_filling = members[members.bd != -99999] # using mean of bd as filling of missing values and outliers
mean_bd = int(tmp_bd_for_filling.bd.mean())
del tmp_bd_for_filling

members['bd'] = members.bd.apply(lambda x: mean_bd if x == -99999 else x )


def bd_zone(x):
	if  x <= 18:
		return 1 # 大學生以下
	elif 18 < x <= 22:
		return 2 # 大學生
	elif 22 < x < 35:
		return 3 # 上班族
	else:
		return 4 # 35以上lol
def city_zone(x):
	if x in set([2,1,20,16,17,17]):
		return 1 # 最不容易流失的city_zone
	elif x in set([11,13,7,18,14,9]):
		return 2 
	elif x in set([10,5,22,6,15,12]):
		return 3 
	else:
		return 4 # 最容易流失的city_zone
def registered_via_zone(x):
	if x in set([4,3]):
		return 1 # 最容易流失的registered_via
	elif x in set([7,2]):
		return 2 # 最不容易流失的registered_via
	else:
		return 3 
members['bd_zone'] = members.bd.apply(bd_zone)
members['city_zone'] = members.city.apply(city_zone)
members['registered_via_zone'] = members.city.apply(registered_via_zone)

#==============================================================================
print('reduce memory')
#==============================================================================
utils.reduce_memory(members)
gc.collect()

# write
path = '../input/preprocessed_data/demographics.csv'


members.to_csv(path, index = False)






