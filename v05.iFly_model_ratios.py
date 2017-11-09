import sys
import csv
import operator
from KS  import compute_KS
from sklearn import linear_model
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

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

def map_emp_length(x):
    if x=='n/a':
        return 1.0
    if '<' in x:
        return 1.0
    if '+' in x:
        return 16.0
    return float(x.split(' ')[0])

#derivative: loan_amnt * APR

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
            el=map_emp_length(line[11])

            fico=float(line[27])
            dti=float(line[24])
            income=float(line[13])
            loan_amount=float(line[2])
            dti2=float(line[2])/float(income)
            apr=float(line[6].replace("%",''))
            revol_bal=float(line[34])
            if ecl<8 or fico<680.0:
                ignored+=1.0
                continue


            ecl=math.log(ecl+1.0)
            tbexm=math.log(float(line[112])+1.0)
            la=math.log(loan_amount+1.0)
            delta_dti=dti-dti2
            lam_apr=apr*la
            tbc_lim=math.log(1.0+float(line[113]))
            revol=math.log(1.0+revol_bal)
            rv2lam=(1.0+float(line[34]))/(float(line[2])+1.0)


            skin=math.log(fico*float(line[112]))

            instalment=float(line[7])


            #my_data.append((float(line[6].replace("%",'')),badGood_mapper[line[16]]))
            X.append([apr,fico,dti,math.log(float(line[13])+1.0),tbc_lim,float(line[93]),msld,ho,pr,float(line[25]),ecl,float(line[107]),tbexm,la,instalment,el])
            Y.append(badGood_mapper[line[16]])

    except:
        ignored+=1
        continue
print "Loans Ignored:", ignored
print "Loans kept:", len(X)
classifier = linear_model.LogisticRegression(C=1e5)
#classifier = GaussianNB()
#classifier= QuadraticDiscriminantAnalysis()
#classifier=SVC(gamma=2.0,C=1.0)

classifier.fit(X,Y)
Z=classifier.predict_proba(X)


my_data=[]
for x in zip(Z,Y):
    my_data.append((x[0][1],x[1]))



print "KS:", compute_KS(my_data)

print "apr,fico,dti,log(anual_income),log(total bc limit),num_accts_120pd,msld,ho,pr,float(line[25]),ecl,float(line[107]),tbexm,la,instalment,el"
my_sorted=sorted([x for x in zip([y[0] for y in Z],X)],key=operator.itemgetter(0),reverse=True)
for x in my_sorted:
    print x
#my_sorted=[x for x in zip(Z,X)]
#for x in my_sorted:
#    print x


    
