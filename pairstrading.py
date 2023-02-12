import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils import optimize
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

# Transaction cost percentage
trcost_perc = 0.0001
MaxIter = 20
CAUSAL = True
ROBUST = True
# m is dim of observations
m_dim = 1
# n is dim of unobserved states
n_dim = 2
# points used to estimate initial coef in linear equation
burn_len = 100
# points used in rolling window of spread estimation
window_size = 20

if ROBUST:
    search_num = 10
else:
    search_num = 1

radi_arr = np.linspace(0.1, 1.0, search_num)
sharpe = np.zeros_like(radi_arr)
sortino = np.zeros_like(radi_arr)

for r_idx in range(search_num):

    radius = radi_arr[r_idx]
    print('Testing radius', radius)

    Y = pd.read_csv('AMZN.csv')
    Y_Close = Y.loc[2012-burn_len:, 'Adj Close'].values
    X = pd.read_csv('GOOG.csv')
    X_Close = X.loc[2012-burn_len:, 'Adj Close'].values


    # transition matrix of unobserved states
    A = np.eye(n_dim)
    # transition matrix of observations
    C = np.zeros((m_dim, n_dim))


    # initial mean and cov of unobserved states
    reg = LinearRegression().fit(X_Close[:burn_len].reshape(burn_len, 1), Y_Close[:burn_len])
    init_mean = np.array([reg.intercept_, reg.coef_[0]]).reshape((n_dim, 1))
    init_cov = np.array([[1.0, 0.0], [0.0, 1.0]])

    # covs
    Bp = np.eye(2)
    Dp = np.array([1.0]).reshape((m_dim, m_dim))


    pre_mean = init_mean.copy()
    pre_cov = init_cov.copy()

    # total time steps, including burn-in periods
    Y_Close = Y_Close[burn_len:]
    X_Close = X_Close[burn_len:]
    horizon = len(Y_Close)
    # observations
    obs = np.zeros((horizon, m_dim))
    # filtered unobserved states
    est_state = np.zeros((horizon, n_dim))
    est_cov = np.zeros((horizon, n_dim, n_dim))

    ################### filtering ######################
    for step in range(horizon):
        obs[step, 0] = Y_Close[step]
        C[0, 0] = 1.0
        C[0, 1] = X_Close[step]
        next_mean, next_cov = optimize(m_dim, n_dim, radius, A, Bp, C, Dp, pre_cov,
                                       obs[step, :].reshape((m_dim, 1)), pre_mean, MaxIter,
                                       causal=CAUSAL, robust=ROBUST)

        est_state[step, :] = next_mean.reshape(-1)
        est_cov[step, :] = next_cov

        pre_mean = next_mean.copy()
        pre_cov = next_cov.copy()
        # if step%5 == 0:
        #     print('Filtered step', step)
        #     print('Estimated det', np.linalg.det(next_cov))
            # if np.linalg.det(next_cov) < 0.0:
            #     print(res.constr_violation,' CG stop cond:', res.cg_stop_cond, 'Status:', res.status)

    estimated = est_state[:, 0] + np.multiply(est_state[:, 1], X_Close)


    ########### trading ##############

    idx = window_size
    spread = Y_Close - estimated

    open_thres = 2.0
    close_thres = 0.0

    position = None
    # 0 for Y, 1 for X
    stock_poi = np.zeros((2, horizon))
    # stock trading volume
    quantity = 100
    cash = np.zeros(horizon)
    cash[:idx] = 10000.0
    stock_value = np.zeros(horizon)

    while idx < horizon:
        roll_mean = spread[idx-window_size:idx].mean()
        roll_std = spread[idx-window_size:idx].std()
        residual = spread[idx] - roll_mean

        if position == None:
            if residual <= open_thres*roll_std and residual >= -open_thres*roll_std:
                # Signal not triggered.
                cash[idx] = cash[idx-1]
            elif residual < -open_thres*roll_std:
                # open long position
                stock_poi[0, idx] = quantity
                stock_poi[1, idx] = -est_state[idx, 1] * quantity
                poi0_sign = np.sign(stock_poi[0, idx])
                poi1_sign = np.sign(stock_poi[1, idx])
                cash[idx] = cash[idx - 1] - \
                            (stock_poi[0, idx] * Y_Close[idx] * (1 + poi0_sign * trcost_perc) +
                             stock_poi[1, idx] * X_Close[idx] * (1 + poi1_sign * trcost_perc))
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
                position = "long"
            elif residual > open_thres*roll_std:
                # open short position
                stock_poi[0, idx] = -quantity
                stock_poi[1, idx] = est_state[idx, 1] * quantity
                poi0_sign = np.sign(stock_poi[0, idx])
                poi1_sign = np.sign(stock_poi[1, idx])
                cash[idx] = cash[idx - 1] - \
                            (stock_poi[0, idx] * Y_Close[idx] * (1 + poi0_sign * trcost_perc) +
                             stock_poi[1, idx] * X_Close[idx] * (1 + poi1_sign * trcost_perc))
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
                position = "short"

        elif position == "long":
            if residual < -close_thres*roll_std:
                # maintain position
                stock_poi[:, idx] = stock_poi[:, idx - 1]
                cash[idx] = cash[idx - 1]
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
            else:
                # close position
                poi0_sign = np.sign(stock_poi[0, idx-1])
                poi1_sign = np.sign(stock_poi[1, idx-1])
                cash[idx] = cash[idx-1] + \
                            stock_poi[0, idx-1] * Y_Close[idx] * (1 - poi0_sign * trcost_perc) + \
                            stock_poi[1, idx-1] * X_Close[idx] * (1 - poi1_sign * trcost_perc)
                stock_poi[:, idx] = 0
                stock_value[idx] = 0
                position = None

        else:
            if residual > close_thres*roll_std:
                # maintain position
                stock_poi[:, idx] = stock_poi[:, idx - 1]
                cash[idx] = cash[idx - 1]
                stock_value[idx] = stock_poi[0, idx] * Y_Close[idx] + stock_poi[1, idx] * X_Close[idx]
            else:
                # close position
                poi0_sign = np.sign(stock_poi[0, idx-1])
                poi1_sign = np.sign(stock_poi[1, idx-1])
                cash[idx] = cash[idx-1] + \
                            stock_poi[0, idx-1] * Y_Close[idx] * (1 - poi0_sign * trcost_perc) + \
                            stock_poi[1, idx-1] * X_Close[idx] * (1 - poi1_sign * trcost_perc)
                stock_poi[:, idx] = 0
                stock_value[idx] = 0
                position = None

        idx += 1



    if ROBUST:
        sub_folder = '{}_{}_{}_{}_{}'.format('causal', CAUSAL, 'robust', ROBUST, round(radius, 2))

        log_dir = './logs/{}'.format(sub_folder)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save params configuration
        with open('{}/params.txt'.format(log_dir), 'w') as fp:
            fp.write('Params setting \n')
            fp.write('COT: {} \n'.format(CAUSAL))
            fp.write('Robust: {} \n'.format(ROBUST))
            fp.write('Bp: {} \n'.format(Bp))
            fp.write('Dp: {} \n'.format(Dp))
            fp.write('init_mean: {} \n'.format(init_mean))
            fp.write('init_cov: {} \n'.format(init_cov))
            fp.write('radius: {} \n'.format(radius))
            fp.write('horizon: {} \n'.format(horizon))
            fp.write('open thres: {} \n'.format(open_thres))
            fp.write('close thres: {} \n'.format(close_thres))
            fp.write('stock trading quantity: {} \n'.format(quantity))
            fp.write('maxiter: {} \n'.format(MaxIter))

        plt.figure(1)
        plt.plot(cash + stock_value, label='total')
        # plt.plot(cash, label='cash')
        # plt.plot(stock_value, label='stock')
        plt.legend(loc='best')
        plt.savefig('{}/portfolio.pdf'.format(log_dir), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)

        plt.figure(2)
        plt.plot(est_state[:, 1], label='hedge ratio')
        plt.legend(loc='best')
        plt.savefig('{}/hedgeratio.pdf'.format(log_dir), format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)

        with open('{}/est_state.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(est_state, fp)

        with open('{}/spread.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(spread, fp)

        with open('{}/cash.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(cash, fp)

        with open('{}/stock_value.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(stock_value, fp)

        with open('{}/position.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(stock_poi, fp)

    ##### Calculate Sharpe and Sortino ratios
    ptf = cash + stock_value
    rtn = np.divide(ptf[1:] - ptf[:-1], ptf[:-1])
    rtn_mean = rtn.mean()
    so_idx = rtn < rtn_mean
    sharpe[r_idx] = (rtn_mean - 0.02/252)/rtn.std()*np.sqrt(252)
    sortino[r_idx] = (rtn_mean - 0.02/252)/rtn[so_idx].std()*np.sqrt(252)

    print('Sharpe ratio of strategy:', sharpe[r_idx])
    print('Sortino ratio of strategy:', sortino[r_idx])

    X_rtn = np.divide(X_Close[1:] - X_Close[:-1], X_Close[:-1])
    X_m = X_rtn.mean()
    X_idx = X_rtn < X_m
    print('Asset 1 Sharpe ratio:', (X_m - 0.02/252)/X_rtn.std()*np.sqrt(252))
    print('Asset 1 Sortino ratio:', (X_m - 0.02/252)/X_rtn[X_idx].std()*np.sqrt(252))

    Y_rtn = np.divide(Y_Close[1:] - Y_Close[:-1], Y_Close[:-1])
    Y_m = Y_rtn.mean()
    Y_idx = Y_rtn < Y_m
    print('Asset 2 Sharpe ratio:', (Y_m - 0.02/252)/Y_rtn.std()*np.sqrt(252))
    print('Asset 2 Sortino ratio:', (Y_m - 0.02/252)/Y_rtn[Y_idx].std()*np.sqrt(252))

print('Sharpe')
print(np.round(sharpe, 4))
print('Sortino')
print(np.round(sortino, 4))

if ROBUST:
    with open('./logs/sharpe_causal_{}.pickle'.format(CAUSAL), 'wb') as fp:
        pickle.dump(sharpe, fp)

    with open('./logs/sortino_causal_{}.pickle'.format(CAUSAL), 'wb') as fp:
        pickle.dump(sortino, fp)

