#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

It took 3.275 hours
'''

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import utils # written by author
from glob import glob
from datetime import datetime
from collections import Counter
from collections import defaultdict

##############
#data preprocessing
##############

# 1.transactions.csv

# os.system('python3 -u transaction_splitting.py') #done 
# os.system('python3 -u train_generation.py') # done
# os.system('python3 -u transaction_date_base.py') # done 
# os.system('python3 -u transaction_price_and_play_days_base.py') # done 
# os.system('python3 -u transaction_time_diff.py') # done # 和上一筆的交易時間差 
# os.system('python3 -u days_since_the_last_transactions.py') # done

#os.system('python3 -u transaction_payment_method.py') # done  # 代表的是what this payment method looks like?

# 2.members.csv

# os.system('python3 -u demographics.py') # done

# 3.user_logs.csv
# os.system('python3 -u user_logs_splitting.py') # done # what the user's listening behavior




