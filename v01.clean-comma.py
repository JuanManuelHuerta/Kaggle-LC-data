import sys
import csv


my_set=map(int,sys.argv[1:])
my_r=csv.reader(sys.stdin)
for line in my_r:
    print [line[x] for x in my_set]

    
