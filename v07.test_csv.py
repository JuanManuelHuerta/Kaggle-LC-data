import sys
import csv


fp=open('aug10loans.csv','rt')
my_r=csv.reader(fp)
for x in my_r:
    print x['member_id']
