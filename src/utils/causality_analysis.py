from statsmodels.tsa.stattools import grangercausalitytests

def granger_causality(data, variables, maxlag=5):
    return grangercausalitytests(data[variables], maxlag=maxlag)