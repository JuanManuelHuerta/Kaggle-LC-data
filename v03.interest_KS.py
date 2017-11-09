import sys
import csv
import operator
from KS  import compute_KS






badGood_mapper={'Charged Off':1, 'Fully Paid':0}
my_r=csv.reader(sys.stdin)
my_data=[]
for line in my_r:
    try:
        if line[16] in badGood_mapper:
            my_data.append((float(line[6].replace("%",'')),badGood_mapper[line[16]]))
    except:
        continue
print "KS:", compute_KS(my_data)


    
