import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from evaluation import plot_key_figures, calc_key_figures

prices = pd.read_csv('prices_train.csv', index_col='AsOfDate', parse_dates=['AsOfDate'])
ret = prices.ffill().diff()

# FOR OPTIMAL SIM:
# cov_historys = 100*np.arange(1,6)
# c_lambs = 0.2*np.arange(0,6)
# lamb = 100
# k = 2
# Results
# optimal cov_history = 300
# optimal c_lamb = 0.2
# optimal sharpe ratio = 0.9657687425798849
# optimal adjusted sharpe ratio = 0.88

#cov_historys = 10 + 50*np.arange(10)
cov_historys = 100*np.arange(6,11)
k = 2 # number of perdictions ahead
lamb = 100
c_lambs = 0.2*np.arange(3)
random_prediction = "linear"

# I have no idea what is good here. Is the code correct??????

# sharp ratio 0.86
# COV_HISTORY = 300 
# reg_history = 300
# k = 2 # number of perdictions ahead
# lamb = 1000
# c_lamb = 0.3

def C_filter(C, c_lamb):
    C_filter = c_lamb*(np.ones(C.shape) - np.eye(C.shape[0])) + np.eye(C.shape[0])
    return C_filter

def covariance(prices, c_lamb, cov_history, t):
    covariance = prices.iloc[-cov_history+t:t,:].cov()
    C_t = covariance.to_numpy()
    mean_diff = (prices.iloc[-cov_history+t:t,:] - prices.iloc[-cov_history+t:t,:].mean()).to_numpy()
    C_t_2 = mean_diff.T @ mean_diff / (mean_diff.shape[0] - 1) # estimate of covariance formula
    assert np.all(np.isclose(C_t, C_t_2))
    C_t_regularized = C_filter(C_t, c_lamb)*C_t
    return C_t_regularized

def covariance_regression(prices, c_lamb, cov_history, t):
    coeffs = linear_regression(prices, cov_history, t)
    X_history = np.stack((np.arange(t-cov_history,t), np.ones(cov_history)), axis=1)
    mean_price_history = X_history @ coeffs
    mean_diff_price_history = prices.iloc[t-cov_history:t,:].to_numpy() - mean_price_history
    C_t = mean_diff_price_history.T @ mean_diff_price_history / (mean_diff_price_history.shape[0] - 1) # covariance matrix
    C_t_regularized = C_filter(C_t, c_lamb)*C_t
    return C_t_regularized

def covariance_rolling(prices, c_lamb, cov_history, t):
    rolling_prices = prices.ewm(com=5).mean()
    X_history = np.stack((np.arange(t-cov_history,t), np.ones(cov_history)), axis=1)
    mean_diff_price_history = prices.iloc[t-cov_history:t,:].to_numpy() - rolling_prices.iloc[t-cov_history:t,:].to_numpy()
    C_t = mean_diff_price_history.T @ mean_diff_price_history / (mean_diff_price_history.shape[0] - 1) # covariance matrix
    C_t_regularized = C_filter(C_t, c_lamb)*C_t
    return C_t_regularized

def calculate_A(prices, t):
    diff = prices.diff()
    diff = diff.to_numpy()
    diff = diff[t+1:t+k+1,:].flatten()
    A = np.diag(diff)
    return A

def calculate_B_et(prices, xm1):
    m = prices.shape[1]
    B = np.eye(m*k) + np.diag(-1*np.ones(k*m-1), -1)
    e_t = np.zeros(k*m)
    e_t[0:m] = xm1
    return B, e_t

def linear_regression(prices, reg_history, t):
    X = np.stack((np.arange(t-reg_history,t), np.ones(reg_history)), axis=1)
    Y = prices.iloc[t-reg_history:t,:].to_numpy() # p_t-(reg_history),...p_t-1 number of values is reg_history.
    assert Y.shape[0] == reg_history
    coeffs = np.linalg.solve(X.T @ X, X.T @ Y)
    return coeffs

def predict(prices, reg_history, t):
    prices_pred = prices.iloc[:t+k+1,:].copy() # at t we know ..., p_t-1. Predict p_t,...,p_t+k (k + 1  values)
    coeffs = linear_regression(prices, reg_history, t)
    X_preds = np.stack((np.arange(t, t+k+1), np.ones(k+1)), axis=1) # predict p*_t,...,p*_t+k
    P_preds = X_preds @ coeffs
    #P_preds = prices.iloc[t-reg_history:t,:].mean()
    prices_pred.iloc[t:,:] = P_preds
    if t == 0:
        plt.figure()
        plt.plot(prices.iloc[:t+k+1,:])
        X_plot =  np.stack((np.arange(t-reg_history, t+k+1), np.ones(k+1+reg_history)), axis=1)
        #Y_plot = X_plot @ coeffs
        #plt.plot(X_plot[:,0], Y_plot)
        plt.plot(prices_pred.iloc[t:,:])
        plt.show()
    return prices_pred

def predict_regression(prices, c_lamb, reg_history, t):
    prices_pred = prices.iloc[:t+k+1,:].copy() # at t we know ..., p_t-1. Predict p_t,...,p_t+k (k + 1  values)
    coeffs = linear_regression(prices, reg_history, t)
    X_preds = np.stack((np.arange(t, t+k+1), np.ones(k+1)), axis=1) # predict p*_t,...,p*_t+k
    # predict mean as trend line
    P_pred_mean = X_preds @ coeffs
    C_t = covariance_regression(prices, c_lamb, reg_history, t)
    P_pred_random = np.random.multivariate_normal(np.zeros(prices.shape[1]), C_t, k+1)
    prices_pred.iloc[t:,:] = P_pred_mean + 0*P_pred_random
    if t == 0:
        plt.figure()
        X_plot =  np.stack((np.arange(t-reg_history, t+k+1), np.ones(k+1+reg_history)), axis=1)
        Y_plot = X_plot @ coeffs
        fit_copy = prices.iloc[t-reg_history:t+k+1,:].copy()
        fit_copy.iloc[:,:] = Y_plot
        plt.plot(prices.iloc[:t+k+1,:])
        plt.plot(fit_copy)
        plt.plot(prices_pred.iloc[t:,:])
        plt.show()
    return prices_pred

def predict_rolling(prices, c_lamb, reg_history, t):
    prices_pred = prices.iloc[:t+k+1,:].copy() # at t we know ..., p_t-1. Predict p_t,...,p_t+k (k + 1  values)
    rolling_prices = prices.ewm(com=5).mean()
    coeffs = linear_regression(prices, reg_history, t)
    X_preds = np.stack((np.arange(t, t+k+1), np.ones(k+1)), axis=1) # predict p*_t,...,p*_t+k
    # predict mean as trend line
    P_pred_mean = X_preds @ coeffs
    # predict deviations as random noise with historical covariance (assume constant over reg_preiod)
    C_t = covariance_rolling(prices, c_lamb, reg_history, t)
    P_pred_random = np.random.multivariate_normal(np.zeros(prices.shape[1]), C_t, k+1)
    prices_pred.iloc[t:,:] = P_pred_mean + 0*P_pred_random
    if t == 0:
        plt.figure()
        X_plot =  np.stack((np.arange(t-reg_history, t+k+1), np.ones(k+1+reg_history)), axis=1)
        Y_plot = X_plot @ coeffs
        fit_copy = prices.iloc[t-reg_history:t+k+1,:].copy()
        fit_copy.iloc[:,:] = Y_plot
        plt.plot(prices.iloc[:t+k+1,:])
        plt.plot(fit_copy)
        plt.plot(rolling_prices.iloc[:t])
        plt.show()
    return prices_pred

def optimize(prices, x_tm1, c_lamb, cov_history, lamb, delta, t, random_prediction):
    m = prices.shape[1]
    x = cp.Variable(m*k)
    A = calculate_A(prices, t) # added 
    B, e_t = calculate_B_et(prices, x_tm1)
    # C_t = covariance(prices, c_lamb, cov_history, t)
    if random_prediction == "linear":
        C_t = covariance_regression(prices, c_lamb, cov_history, t)
    elif random_prediction == "rolling":
        C_t = covariance_rolling(prices, c_lamb, cov_history, t)
    else:
        C_t = covariance(prices, c_lamb, cov_history, t)
    C = np.kron(np.eye(k), C_t)
    objective = cp.Minimize( x.T @ cp.psd_wrap(C) @ x - lamb*cp.sum(A @ x - delta*cp.abs(B @ x - e_t)) )
    constraints = []
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=cp.MOSEK)
    if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
    return  x.value[0:m] # output optimal position!

def trend_model(prices, c_lamb, cov_history, lamb, random_prediction):
    reg_history = cov_history # ideally, reg and cov history should be equal.
    ret = prices.ffill().diff()
    pos = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    delta = 0.0002
    x_tm1 = np.zeros(prices.shape[1])
    # loop over all dates
    #for t in range(cov_history, cov_history+1000):#prices.shape[0]-k):
    for t in range(cov_history, prices.shape[0]-k):
        #if t % 50 == 0:
        #    print(t)
        ### predict here k time steps ahead of t.
        #prices_pred = predict(prices, reg_history, t)
        if random_prediction == "linear":
            prices_pred = predict_regression(prices, c_lamb, reg_history, t)
        elif random_prediction == "rolling":
            prices_pred = predict_rolling(prices, c_lamb, reg_history, t)
        else:
            prices_pred = predict(prices, reg_history, t)
        x_t = optimize(prices_pred, x_tm1, c_lamb, cov_history, lamb, delta, t, random_prediction)
        x_tm1 = x_t
        pos.iloc[t,:] = x_t
    return pos




sharp_ratios = np.zeros((cov_historys.size, c_lambs.size))
sharp_max = -np.inf
for i in range(cov_historys.size):
    for j in range(c_lambs.size): # x-axis
        print((i,j))
        pos = trend_model(prices, c_lambs[j], cov_historys[i], lamb, random_prediction)
        results = calc_key_figures(pos, prices)
        sharp_ratios[i,j] = results['sharpe']
        if results['sharpe'] > sharp_max:
            c_lamb_max = c_lambs[j]
            cov_hist_max = cov_historys[i]
            sharp_max = results['sharpe']
            pos_max = pos

print('optimal cov_history =',  cov_hist_max)
print('optimal c_lamb =',  c_lamb_max)
print('optimal sharpe ratio =',  sharp_max)
pos_max.to_csv('optimal_pos.csv')

if cov_historys.size > 1 and c_lambs.size > 1:
    fig, ax = plt.subplots()
    CS = ax.contour(c_lambs, cov_historys, sharp_ratios)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Sharpe Ratio - Linear')
    ax.grid()
    plt.savefig('linear2.pdf')
    plt.show()
elif cov_historys.size > 1 and c_lambs.size == 1:
    fig, ax = plt.subplots()
    CS = ax.plot(cov_historys, sharp_ratios)
    ax.set_title('Sharpe Ratio')
    ax.grid()
    plt.show()
elif cov_historys.size == 1 and c_lambs.size > 1:
    fig, ax = plt.subplots()
    CS = ax.plot(c_lambs, sharp_ratios)
    ax.set_title('Sharpe Ratio')
    ax.grid()
    plt.show()

plot_key_figures(pos_max, prices)
plt.savefig('linear_600_02.pdf')
plt.show()

#cost_coeffs = np.linspace(0, -0.1, 10)
#sharpe_per_cost_coef = [calc_key_figures(pos, prices, costs=cost_coef, key_figures=['sharpe'])['sharpe'] for cost_coef in cost_coeffs]

#plt.figure()
#plt.plot(cost_coeffs, sharpe_per_cost_coef)
#plt.title('Cost sensitivity')
#plt.ylabel('Sharpe after costs')
#plt.xlabel('Cost cofficient')
#plt.grid()
#plt.show()


#model_ret = (pos.shift(1)*ret).sum(axis=1)
#plt.figure()
#model_ret.cumsum().plot()
#plt.grid()
#plt.title('Cumulative returns excluding costs')
#plt.show()

#pos = trend_model(ret)
#pos_short = trend_model(ret.iloc[:-20])
#(pos-pos_short).abs().sum()

def run():
    t = ""
    with open('simulation.py') as f:
        t = f.read()
    return t