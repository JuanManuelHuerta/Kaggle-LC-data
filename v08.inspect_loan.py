import sys
import csv

fp=sys.stdin
headline=fp.readline().strip().split(",")
loan=sys.argv[1]
for line in csv.reader(fp):
    if line[0]==loan:
        z=[w for w in zip(headline,line)]
        print z


