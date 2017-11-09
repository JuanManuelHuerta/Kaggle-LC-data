import sys
import csv
import operator
from KS  import compute_KS
from sklearn import linear_model
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
import pickle
from iFly import *

##  Load data
##  Fit model
##  Score
##  KS

##  27 fico low
##24 dti
## 13  annual_inc
## 113 total_bc_limit
##93 um_accts_ever_120_pd
#6  APR
#16 msld months since las delinquency
## 33 pr public record
## ho Rent or not
## 25 delinq_2yrs
##26 earliest_cr_line
#107:"pct_tl_nvr_dlq"
#77: inq_last_12m
#112:total_bal_ex_mort
#2:loan_amnt
#34:"revol_bal"
#7: instalment
#11: emp_length


'''
1:"member_id"
2:"loan_amnt"
3:"funded_amnt"
4:"funded_amnt_inv"
5:"term"
6:"int_rate"
7:"installment"
8:"grade"
9:"sub_grade"
10:"emp_title"
11:"emp_length"
12:"home_ownership"
13:"annual_inc"
14:"verification_status"
15:"issue_d"
16:"loan_status"
17:"pymnt_plan"
18:"url"
19:"desc"
20:"purpose"
21:"title"
22:"zip_code"
23:"addr_state"
24:"dti"
25:"delinq_2yrs"
26:"earliest_cr_line"
27:"fico_range_low"
28:"fico_range_high"
29:"inq_last_6mths"
30:"mths_since_last_delinq"
31:"mths_since_last_record"
32:"open_acc"
33:"pub_rec"
34:"revol_bal"
35:"revol_util"
36:"total_acc"
37:"initial_list_status"
38:"out_prncp"
39:"out_prncp_inv"
40:"total_pymnt"
41:"total_pymnt_inv"
42:"total_rec_prncp"
43:"total_rec_int"
44:"total_rec_late_fee"
45:"recoveries"
46:"collection_recovery_fee"
47:"last_pymnt_d"
48:"last_pymnt_amnt"
49:"next_pymnt_d"
50:"last_credit_pull_d"
51:"last_fico_range_high"
52:"last_fico_range_low"
53:"collections_12_mths_ex_med"
54:"mths_since_last_major_derog"
55:"policy_code"
56:"application_type"
57:"annual_inc_joint"
58:"dti_joint"
59:"verification_status_joint"
60:"acc_now_delinq"
61:"tot_coll_amt"
62:"tot_cur_bal"
63:"open_acc_6m"
64:"open_il_6m"
65:"open_il_12m"
66:"open_il_24m"
67:"mths_since_rcnt_il"
68:"total_bal_il"
69:"il_util"
70:"open_rv_12m"
71:"open_rv_24m"
72:"max_bal_bc"
73:"all_util"
74:"total_rev_hi_lim"
75:"inq_fi"
76:"total_cu_tl"
77:"inq_last_12m"
78:"acc_open_past_24mths"
79:"avg_cur_bal"
80:"bc_open_to_buy"
81:"bc_util"
82:"chargeoff_within_12_mths"
83:"delinq_amnt"
84:"mo_sin_old_il_acct"
85:"mo_sin_old_rev_tl_op"
86:"mo_sin_rcnt_rev_tl_op"
87:"mo_sin_rcnt_tl"
88:"mort_acc"
89:"mths_since_recent_bc"
90:"mths_since_recent_bc_dlq"
91:"mths_since_recent_inq"
92:"mths_since_recent_revol_delinq"
93:"num_accts_ever_120_pd"
94:"num_actv_bc_tl"
95:"num_actv_rev_tl"
96:"num_bc_sats"
97:"num_bc_tl"
98:"num_il_tl"
99:"num_op_rev_tl"
100:"num_rev_accts"
101:"num_rev_tl_bal_gt_0"
102:"num_sats"
103:"num_tl_120dpd_2m"
104:"num_tl_30dpd"
105:"num_tl_90g_dpd_24m"
106:"num_tl_op_past_12m"
107:"pct_tl_nvr_dlq"
108:"percent_bc_gt_75"
109:"pub_rec_bankruptcies"
110:"tax_liens"
111:"tot_hi_cred_lim"
112:"total_bal_ex_mort"
113:"total_bc_limit"
114:"total_il_high_credit_limit"
'''

badGood_mapper={'Charged Off':1, 'Fully Paid':0}



#derivative: loan_amnt * APR

X=[]
Y=[]
print "Training set:", sys.argv[1]

fp=open(sys.argv[1])
my_r=csv.reader(fp)
header_1=fp.readline().rstrip().split(",")
header_mapper_1={}
for x in zip(range(len(header_1)),header_1):
    header_mapper_1[x[1].replace("\"","")]=x[0]

my_data=[]
ignored=1
for line in my_r:
    try:
        x,y = featurizer(line,header_mapper_1)
        if x and x:
            X.append(x[1])
            Y.append(y)
        else:
            ignored+=1

    except:
        ignored+=1
        continue

print "Train Loans Ignored:", ignored
print "Train Loans kept:", len(X)
classifier = linear_model.LogisticRegression(C=1e5,max_iter=500)
#classifier = GaussianNB()
#classifier= QuadraticDiscriminantAnalysis()
#classifier=SVC(gamma=2.0,C=1.0)

classifier.fit(X,Y)



print "On development set:"

Z=classifier.predict_proba(X)
my_data=[]
for x in zip(Z,Y):
    my_data.append((x[0][1],x[1]))
print "KS:", compute_KS(my_data)



pickle.dump(classifier,open(sys.argv[2],"wb"))    


