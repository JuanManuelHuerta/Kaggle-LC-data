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

def map_emp_length(x):
    if x=='n/a':
        return 1.0
    if '<' in x:
        return 1.0
    if '+' in x:
        return 16.0
    return float(x.split(' ')[0])

def map_emp_length_api(x):
    if x == None:
        return 16.0
    if type(x) is int:
        x2= x/16.0
        if x<1.0:
            return 1.0
        if x> 16.0:
            return 16.0
        else:
            return x
    return x



def RepresentsInt(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False



def featurizer(line,header_mapper,return_loan_status=True,api_call=False):
    try:
        
        if  (return_loan_status==True and line[header_mapper["loan_status"]] in badGood_mapper) or (return_loan_status==False):

            if return_loan_status==True:
                y =badGood_mapper[line[header_mapper["loan_status"]]]



            msld=60



            if RepresentsInt(line[header_mapper["mths_since_last_delinq"]])==True:
                if float(line[header_mapper["mths_since_last_delinq"]])<60:
                    msld=float(line[header_mapper["mths_since_last_delinq"]])

            ho=0
            if line[header_mapper["home_ownership"]]=="RENT":
                ho=1




            pr=0
            if RepresentsInt(line[header_mapper["pub_rec"]])==True:
                pr=float(line[header_mapper["pub_rec"]])



            if RepresentsInt(line[header_mapper["earliest_cr_line"]].split("-")[1])==True:
                ecl=2016-float(line[header_mapper["earliest_cr_line"]].split("-")[1])


            if api_call==False:
                el=map_emp_length(line[header_mapper["emp_length"]])
            else:
                el=map_emp_length_api(line[header_mapper["emp_length"]])



            #fico=float(line[27])
            fico=float(line[header_mapper["fico_range_low"]])
            #if line[header_mapper["bc_open_to_buy"]]!="":
            #    bcotb=math.log(1.0+float(line[header_mapper["bc_open_to_buy"]]))
            #else:
            #    bcotb=0.0
            if line[header_mapper["mths_since_recent_inq"]]!="":
                msri=float(line[header_mapper["mths_since_recent_inq"]])
            else:
                msri=999.0

            if line[header_mapper["revol_util"]]!="":
                r_u=float(line[header_mapper["revol_util"]].replace("%",""))
            else:
                r_u=0.0


            dti=float(line[header_mapper["dti"]])
            income=float(line[header_mapper["annual_inc"]])

            if "loan_amnt" in header_mapper:
                loan_amount=float(line[header_mapper["loan_amnt"]])
                dti2=float(line[header_mapper["loan_amnt"]])/float(income)
            else:
                loan_amount=float(line[header_mapper["loan_amount"]])
                dti2=float(line[header_mapper["loan_amount"]])/float(income)
                

            if type(line[header_mapper["int_rate"]]) is str:
                apr=float(line[header_mapper["int_rate"]].replace("%",''))
            else:
                apr=float(line[header_mapper["int_rate"]])

            revol_bal=float(line[header_mapper["revol_bal"]])



            if ecl<8 or fico<680.0:
                ignored+=1.0
                pass


            ecl=math.log(ecl+1.0)
            tbexm=math.log(float(line[header_mapper["total_bal_ex_mort"]])+1.0)



            la=math.log(loan_amount+1.0)




            delta_dti=dti-dti2
            lam_apr=apr*la





            tbc_lim=math.log(1.0+float(line[header_mapper["total_bc_limit"]]))
            revol=math.log(1.0+revol_bal)





            if "loan_amnt" in header_mapper:

                rv2lam=(1.0+float(line[header_mapper["revol_bal"]]))/(float(line[header_mapper["loan_amnt"]])+1.0)
                rv2lam=(float(line[header_mapper["revol_bal"]]))-(float(line[header_mapper["loan_amnt"]]))

            else:


                rv2lam=(1.0+float(line[header_mapper["revol_bal"]]))/(float(line[header_mapper["loan_amount"]])+1.0)
                rv2lam=(float(line[header_mapper["revol_bal"]]))-(float(line[header_mapper["loan_amount"]]))



            rv2lam=math.copysign(math.log(math.fabs(rv2lam)) if rv2lam!=0.0 else 0.0,rv2lam)
            skin=math.log(fico*float(line[header_mapper["total_bal_ex_mort"]]))




            instalment=float(line[header_mapper["installment"]])

            ## ECL is wrong  TO DO!!!! the number _ prefixes is a mess
            ###print "Still here", [apr,fico,dti,math.log(float(line[header_mapper["annual_inc"]])+1.0),tbc_lim]
            ###print [float(line[header_mapper["num_accts_ever_120_pd"]]),msld,ho,pr]
            #print [float(line[header_mapper["delinq_2yrs"]]),ecl,float(line[header_mapper["pct_tl_nvr_dlq"]])], [tbexm,la,instalment,el,rv2lam]



            #my_data.append((float(line[header_mapper["int_rate"]].replace("%",'')),badGood_mapper[line[16]]))
            x=[apr,fico,dti,math.log(float(line[header_mapper["annual_inc"]])+1.0),tbc_lim,float(line[header_mapper["num_accts_ever_120_pd"]]),msld,ho,pr,float(line[header_mapper["delinq_2yrs"]]),ecl,float(line[header_mapper["pct_tl_nvr_dlq"]]),tbexm,la,instalment,el,rv2lam,msri,r_u]





            if return_loan_status==True:
                return [line[header_mapper['id']],x], y
            else:
                return [line[header_mapper['id']],x]
        return
    except:
        return


def json2array_header(x_in):
    x=[]
    y={}
    i=0
    for key in x_in:
        x.append(x_in[key])
        y[key]=i
        i+=1
    return x, y
