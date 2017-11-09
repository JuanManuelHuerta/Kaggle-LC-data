import sys
import csv


#my_set=map(int,sys.argv[1:])
my_r=csv.reader(sys.stdin)
header=sys.stdin.readline()
my_w=csv.writer(sys.stdout)
sys.stdout.write(header)


for line in my_r:
    try:
        if line[16]=='Charged Off' or line[16]=='Fully Paid':
        #print [line[x] for x in my_set]
            my_w.writerow(line)
    except:
        continue


    
