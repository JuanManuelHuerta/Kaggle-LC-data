import sys
import csv
import operator
from KS  import compute_KS
from sklearn import linear_model
import math


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

X=[]
Y=[]
badGood_mapper={'Charged Off':1, 'Fully Paid':0}
my_r=csv.reader(sys.stdin)
my_data=[]
ignored=1
for line in my_r:
    try:
        if line[16] in badGood_mapper:
            msld=60
            if line[30]!="":
                if float(line[30])<60:
                    msld=float(line[30])
            ho=0
            if line[12]=="RENT":
                ho=1
            pr=0
            if line[33]!="":
                pr=float(line[33])
            ecl=0.0
            if line[26]!="":
                ecl=2016-float(line[26].split("-")[1])
            ecl=math.log(ecl+1.0)
            tbexm=math.log(float(line[112])+1.0)

            la=math.log(float(line[2])+1.0)
            #my_data.append((float(line[6].replace("%",'')),badGood_mapper[line[16]]))
            X.append([float(line[27]),float(line[24]),math.log(float(line[13])+1.0),math.log(1.0+float(line[113])),float(line[93]),float(line[6].replace("%",'')),msld,ho,pr,float(line[25]),ecl,float(line[107]),tbexm,la])
            Y.append(badGood_mapper[line[16]])

    except:
        ignored+=1
        continue
print "Loans Ignored:", ignored
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X,Y)
Z=logreg.predict_proba(X)

my_data=[]
for x in zip(Z,Y):
    my_data.append((x[0][1],x[1]))


print "KS:", compute_KS(my_data)


    
