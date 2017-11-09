import sys
import operator
import math

def compute_KS(x):
    s_sorted=sorted(x,key=operator.itemgetter(0))
    n_goods=float(sum(1.0  for y in x if y[1]==0))
    n_bads=float(sum(1.0  for y in x if y[1]==1))
    ks=0.0
    accum=[0.0,0.0]
    for i in range(len(s_sorted)):
        accum[s_sorted[i][1]]+=1.0
        current=math.fabs(accum[0]/n_goods-accum[1]/n_bads)
        if current>ks:
            ks=current
    return ks


