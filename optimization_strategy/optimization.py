import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import cvxpy as cp

def covariance(price_p_pred, t):
    ''''
    covariance computes ...
    Input: 
    Output:
    '''   
    covariance = prices.iloc[-COV_HISTORY:,:].cov()
    
    covariance.to_numpy()
    return 

def calculate_A(price_p_pred, t):
    '''
    [[P_1t, P_1(t+1), ..., P_,(t+k)],
    [P_2t, P_2(t+1)],
    ...,
    P_mt, P_m(t+1)]
    '''
    
    diff = price_p_pred.diff()
    diff[t+1:-1,:].flatten()
    A = np.diag(diff)
    return A

def calculate_B(price_p_pred, delta, history):
    (k, m) = np.size(price_p_pred)
    B = np.eye(m*k) + np.diag(-1*np.ones(k*m-1),-1)
    
    e_t = history[-1,:]
    return -1

def create_et():
    

COV_HISTORY = 100
ASSET_HISTORY = np.array()

def main():  
    price_p_pred = np.array([[1,2,3,4],
                            [5,6,7,8],
                            [9,10,11,12]]
                            )
    t = 3
    calculate_A(price_p_pred, t-1)
    
    k = 100
    m = 100
    P = np.ones((k,m))
    np.eye()




def brutto_return(P, prices):
    ''''
    ... computes ...
    Input: 
    Output:
    '''
    return -1

def slippage(P):
    ''''
    ... computes ...
    Input: 
    Output:
    '''
    return -1

def optimize():
    ''''
    ... computes ...
    Input: 
    Output:
    '''
    return -1 