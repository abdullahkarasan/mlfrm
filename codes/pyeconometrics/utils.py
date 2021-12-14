# -*- coding: utf-8 -*-
from math import sqrt, factorial, exp
import scipy.stats as st


def norm_cdf(x):
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)/sqrt(2.0)
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x)
    return 0.5*(1.0 + sign*y)
    
def unique_permutations(seq):
    i_indices = range(len(seq)-1, -1, -1)
    k_indices = i_indices[1:]
    seq = sorted(seq)
    while True:
        yield seq
        for k in k_indices:
            if seq[k] < seq[k+1]:
                break
        else:
            return
        k_val = seq[k]
        for i in i_indices:
            if k_val < seq[i]:
                break
        (seq[k], seq[i]) = (seq[i], seq[k])
        seq[k+1:] = seq[-1:k:-1]

def nCr(n,r):
    try:
        return factorial(n) / factorial(r) / factorial(n-r)
    except:
        return 101


def inverse_mills_ratio(x):
    return st.norm.pdf(x) / st.norm.cdf(x)


def derivate_inverse_mills_ratio(x):
    return - inverse_mills_ratio(x) * (x + inverse_mills_ratio(x))