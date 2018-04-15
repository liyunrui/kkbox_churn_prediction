#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri Dec 1 22:22:35 2017

@author: Ray

# create payment_method_feature: It determine what the payment_method looks like?
'''

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import utils # written by author
from glob import glob
from datetime import datetime
from collections import Counter
from collections import defaultdict
##################################################
# Load transaction 
##################################################

transactions = utils.read_multiple_csv('../input/preprocessed_data/transactions')

##################################################
# payment_method_id
##################################################

payment_method_id_count = Counter(transactions['payment_method_id']).most_common()

#core
payment_method = pd.DataFrame({'payment_method_id':[i[0] for i in payment_method_id_count],
                              'count': [i[1] for i in payment_method_id_count]
                              })[['payment_method_id','count']]
payment_method['method_ratio'] = payment_method['count'] / sum(payment_method['count']) 
payment_method['top_3_payment_method'] = [0 if c <1819465 else 1 for c in payment_method['count'] ]
payment_method['between_3_to_5_payment_method'] = [1 if 182139 < c <1819465 else 0 for c in payment_method['count'] ]
payment_method['out_of_10_payment_method'] = [1 if c <247463 else 0 for c in payment_method['count'] ]


##################################################
# plan_list_price
##################################################

plan_list_price_count = Counter(transactions['plan_list_price']).most_common()

#core
payment_price = pd.DataFrame(
                        {'plan_list_price':[i[0] for i in plan_list_price_count],
                              'count': [i[1] for i in plan_list_price_count]
                              }

                            )[['plan_list_price','count']]
payment_price['plan_list_price_ratio'] = payment_price['count'] / sum(payment_price['count'])


pay_m = defaultdict(list)
p_id_bk = None # p_id : payment_method_id
for p_id,p_price in tqdm(transactions[['payment_method_id','plan_list_price']].values[:,:]):
    #if p_price in pay_m[p_id]:
    if p_id_bk == None:
        pass
    else:
        if p_price not in pay_m[p_id]:
            pay_m[p_id].append(p_price)
    p_id_bk = p_id 

pay_m_ = pd.DataFrame({'payment_method_id':[k for k,v in pay_m.items()],
                       'flexible_price':[v for k,v in pay_m.items()]})[['payment_method_id','flexible_price']]
pay_m_['price_len'] = pay_m_.flexible_price.map(len)
pay_m_['price-mean'] = pay_m_.flexible_price.map(np.mean)
pay_m_['price-median'] = pay_m_.flexible_price.map(np.median)
pay_m_['price-max'] = pay_m_.flexible_price.map(max)
pay_m_['price-min'] = pay_m_.flexible_price.map(min)
pay_m_['price_len_ratio'] = pay_m_.price_len / sum(pay_m_.price_len)
pay_m_['is_non-flexible_price'] = [1.0 if i == 1 else 0.0 for i in pay_m_.price_len]


del pay_m


##################################################
# payment_plan_days
##################################################

payment_plan_days_count = Counter(transactions['payment_plan_days']).most_common()



pay_m_days = defaultdict(list) # key: method, value: corresponding list of payment_plan_days 
p_id_bk = None # p_id : payment_method_id
for p_id,p_p_days in tqdm(transactions[['payment_method_id','payment_plan_days']].values[:,:]):
    #if p_price in pay_m[p_id]:
    if p_id_bk == None:
        pass
    else:
        if p_p_days not in pay_m_days[p_id]:
            pay_m_days[p_id].append(p_p_days)
    p_id_bk = p_id 

pay_m_days_ = pd.DataFrame({'payment_method_id':[k for k,v in pay_m_days.items()],
                       'flexible_plan_days':[v for k,v in pay_m_days.items()]})[['payment_method_id','flexible_plan_days']]
pay_m_days_['plan_days_len'] = pay_m_days_.flexible_plan_days.map(len)
pay_m_days_['plan_days-mean'] = pay_m_days_.flexible_plan_days.map(np.mean)
pay_m_days_['plan_days-median'] = pay_m_days_.flexible_plan_days.map(np.median)
pay_m_days_['plan_days-max'] = pay_m_days_.flexible_plan_days.map(max)
pay_m_days_['plan_days-min'] = pay_m_days_.flexible_plan_days.map(min)
pay_m_days_['plan_days_len_ratio'] = pay_m_days_.plan_days_len / sum(pay_m_days_.plan_days_len) # 值越大代表這個方案變動程度很高
pay_m_days_['is_non-flexible_plan_days'] = [1.0 if i == 1 else 0.0 for i in pay_m_days_.plan_days_len] # 1.0 代表此方案一直都是只有固定的日子 and vice versa

del pay_m_days

##################################################
# actual_amount_paid
##################################################

actual_amount_paid_count = Counter(transactions['actual_amount_paid']).most_common()

# core
pay_m_actual_paid = defaultdict(list) # key: method, value: corresponding list of payment_plan_days 
p_id_bk = None # p_id : payment_method_id
for p_id,a_paid in tqdm(transactions[['payment_method_id','actual_amount_paid']].values[:,:]):
    #if p_price in pay_m[p_id]:
    if p_id_bk == None:
        pass
    else:
        if a_paid not in pay_m_actual_paid[p_id]:
            pay_m_actual_paid[p_id].append(a_paid)
    p_id_bk = p_id 

pay_m_actual_paid_ = pd.DataFrame({'payment_method_id':[k for k,v in pay_m_actual_paid.items()],
                       'flexible_actual_paid':[v for k,v in pay_m_actual_paid.items()]})[['payment_method_id','flexible_actual_paid']]
#為什麼會有flexible_actual_paid?可能因為某些方案會有優惠折扣(參見https://ssl.kkbox.com/tw/billing/index.php)
pay_m_actual_paid_['actual_paid_len'] = pay_m_actual_paid_.flexible_actual_paid.map(len)
pay_m_actual_paid_['actual_paid-mean'] = pay_m_actual_paid_.flexible_actual_paid.map(np.mean)
pay_m_actual_paid_['actual_paid-median'] = pay_m_actual_paid_.flexible_actual_paid.map(np.median)
pay_m_actual_paid_['actual_paid-max'] = pay_m_actual_paid_.flexible_actual_paid.map(max)
pay_m_actual_paid_['actual_paid-min'] = pay_m_actual_paid_.flexible_actual_paid.map(min)
pay_m_actual_paid_['actual_paid_len_ratio'] = pay_m_actual_paid_.actual_paid_len / sum(pay_m_actual_paid_.actual_paid_len) # 值越大代表這個方案變動程度很高
pay_m_actual_paid_['is_not_discountable'] = [1.0 if i == 1 else 0.0 for i in pay_m_actual_paid_.actual_paid_len] # 1.0 代表此方案基本上不會有優惠折扣 and vice versa

del pay_m_actual_paid
##################################################
# merge payment_method into payment_plan_days and plan_list_price
##################################################

df = pd.merge(payment_method, pay_m_, 
        on = 'payment_method_id', how = 'left').merge(pay_m_days_,
        on = 'payment_method_id', how = 'left').merge(pay_m_actual_paid_,
        on = 'payment_method_id', how = 'left')

##################################################
# features of payment_method
##################################################

def is_single_purchase(x):
    # 這個方案是單筆購買麼？
    # case1: payment_plan_days == 410 and plan_list_price == 1788
    # case2: payment_plan_days == 195 and plan_list_price == 894
    # case3: payment_plan_days == 180 and plan_list_price == 536 --> 1.國泰世華KOKO(COMBO)信用卡 2.國泰世華Play悠遊聯名卡
    if 1788 in x.flexible_price and 410 in x.flexible_plan_days:
        return 1.0
    elif 894 in x.flexible_price and 195 in x.flexible_plan_days:
        return 1.0
    elif 536 in x.flexible_price and 180 in x.flexible_plan_days:
        return 1.0
    else:
        return 0.0   
    return
def is_from_cathay(x):
    # 這個方案是刷國泰世華的信用卡麼？
    if 536 in x.flexible_price and 180 in x.flexible_plan_days:
        return 1.0
    else:
        return 0.0
def is_automatic_renewal(x):
    # 這個方案是自動續約麼
    # case1: payment_plan_days = 30, plan_list_price == 149 and actual_amount_paid == 149 --> 1.月租型 2. 中華電信emome付款 
    # case2: payment_plan_days = 30, plan_list_price == 149 and actual_amount_paid <= 99  --> 中國信託酷玩卡優惠（次月回饋50)
    # case3: payment_plan_days = 90, plan_list_price == 298 --> VISA金融卡優惠(第91天起每月自動扣繳 NT$149)
    # Reference : https://ssl.kkbox.com/tw/billing/index.php
    if 149 in x.flexible_price and 30 in x.flexible_plan_days:
        return 1.0
    elif 298 in x.flexible_price and 90 in x.flexible_plan_days:
        return 1.0
    else:
        return 0.0    
    return 
def is_visa_debit(x):
    # 這個方案來自VISA金融卡優惠？
    if 298 in x.flexible_price and 90 in x.flexible_plan_days:
        return 1.0
    else:
        return 0.0
def is_from_ctbc_bank(x):
    # 這個方案來自中國信託酷玩卡優惠麼？
    if 149 in x.flexible_price and 30 in x.flexible_plan_days and 99 in x.flexible_actual_paid:
        return 1.0
    else:
        return 0.0
def is_student(x):
    # 這個方案是學生方案麼
    # case0: plan_list_price == 100 -->青年學生限時專案
    # The following is 青年學生專案
    # case1: payment_plan_days = 90, plan_list_price == 300
    # case2: payment_plan_days = 180, plan_list_price == 600
    # case3: payment_plan_days = 360, plan_list_price == 1200
    # case4: Reference: https://www.ptt.cc/bbs/Lifeismoney/M.1485076197.A.0A4.html
    
    # x.flexible_price: list of price 
    # x.flexible_plan_days: list of payment
    if 100 in x.flexible_price:
        return 1.0
    elif 300 in x.flexible_price and 90 in x.flexible_plan_days:
        return 1.0
    elif 600 in x.flexible_price and 180 in x.flexible_plan_days:
        return 1.0
    elif 1200 in x.flexible_price and 360 in x.flexible_plan_days:
        return 1.0
    else:
        return 0.0
def is_free(x):
    # 這個方案曾經有0元麼
    if 0 in x.flexible_price:
        return 1.0
    else:
        return 0.0
        
# core
df['is_student_programe'] = df.apply(is_student, axis = 1)
df['is_automatic_renewal'] = df.apply(is_automatic_renewal, axis = 1)
df['is_single_purchase'] = df.apply(is_single_purchase, axis = 1)
df['is_from_cathay'] = df.apply(is_from_cathay, axis = 1)
df['is_from_ctbc_bank'] = df.apply(is_from_ctbc_bank, axis = 1)
df['is_visa_debit'] = df.apply(is_visa_debit, axis = 1)
df['has_been_free'] = df.apply(is_free, axis = 1)



df.to_csv('../input/preprocessed_data/payment_method.csv')



##################################################
# payment_plan_days ratio
##################################################


payment_plan_days = pd.DataFrame(
                        {'payment_plan_days':[i[0] for i in payment_plan_days_count],
                              'count': [i[1] for i in payment_plan_days_count]
                              }

                            )[['payment_plan_days','count']]
payment_plan_days['payment_plan_days_ratio'] = payment_plan_days['count'] / sum(payment_plan_days['count'])

payment_plan_days.to_csv('../input/preprocessed_data/payment_plan_days_ratio.csv')

##################################################
# plan_list_price ratio
##################################################

payment_price = pd.DataFrame(
                        {'plan_list_price':[i[0] for i in plan_list_price_count],
                              'count': [i[1] for i in plan_list_price_count]
                              }

                            )[['plan_list_price','count']]
payment_price['plan_list_price_ratio'] = payment_price['count'] / sum(payment_price['count'])

payment_price.to_csv('../input/preprocessed_data/plan_list_price_ratio.csv')

##################################################
# actual_amount_paid ratio
##################################################


actual_amount_paid = pd.DataFrame(
                        {'actual_amount_paid':[i[0] for i in actual_amount_paid_count],
                              'count': [i[1] for i in actual_amount_paid_count]
                              }

                            )[['actual_amount_paid','count']]
actual_amount_paid['actual_amount_paid_ratio'] = actual_amount_paid['count'] / sum(actual_amount_paid['count'])


actual_amount_paid.to_csv('../input/preprocessed_data/actual_amount_paid_ratio.csv')

