from urllib2 import Request, urlopen, URLError
import json
import unicodedata
import re
from iFly import *
import unicodedata
import pickle
from sklearn import linear_model
import operator
import sys

def my_lower(x):
    return "_"+x.group(1).lower()

def my_number(x):
    return "_"+x.group(1)



request = Request('https://api.lendingclub.com/api/investor/v1/loans/listing?showAll=true',headers={"Authorization" : "uMX+7pD/8QgU1OLF81H/xK0xzx4="})

try:
    response = urlopen(request)
    available_loans = json.load(response)

except URLError, e:
    print 'Got an error code:', e

print "Accessed through API", len(available_loans['loans']), " loans"
classifier=pickle.load(open(sys.argv[1],"rb"))
print "Loaded Model:", sys.argv[1]

ignored=0
X_eval=[]
X_eval_id=[]
for loan in available_loans['loans']:
    #print loan
    try:
        tmp_dict={}
        for key in loan:
            nk=unicodedata.normalize('NFKD',key).encode('ascii','ignore')
            nk=re.sub(r"([A-Z])",my_lower,nk)
            nk=re.sub(r"([0-9]+)",my_number,nk)
            if isinstance(loan[key],unicode):
                nv=unicodedata.normalize('NFKD',loan[key]).encode('ascii','ignore')
            else:
                nv=loan[key]
            tmp_dict[nk]=nv
        line, header_mapper_2 = json2array_header(tmp_dict)
        print "DEBUG\n", tmp_dict, "\n", header_mapper_2, "\n", line, "\n"
        x  = featurizer(line,header_mapper_2,False,True)
        if x:
            X_eval.append(x[1])
            X_eval_id.append(x[0])
        else:
            ignored+=1
    except:
        ignored+=1
        continue

print "API Loans Ignored:", ignored
print "APIl Loans kept:", len(X_eval)



print "On eval data:"
Z=classifier.predict_proba(X_eval)
print "apr,fico,dti,log(anual_income),log(total bc limit),num_accts_120pd,msld,ho,pr,float(line[25]),ecl,float(line[107]),tbexm,la,instalment,el"
my_sorted=sorted([x for x in zip(X_eval_id,[y[0] for y in Z],X_eval)],key=operator.itemgetter(1),reverse=True)

