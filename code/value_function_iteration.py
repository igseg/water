import pandas as pd
import numpy  as np
from nls_optimizer import *
import statsmodels.api as sm
from data_tools import load_data
from Models.models import *
##################### Initialise parameters #####################

from parameters import *

# ts = load_data()
#
# ts['water_prices'] = np.zeros(ts.shape[0])
# ts.price_power = ts.price_power / 100
# ts = ts.set_index(pd.to_datetime(ts.index.values))
#
# prev_time = pd.Timestamp(ts.index.values[-1])
# print(ts.loc[prev_time,  'water_prices'])
#
# ##### Set Water Price Level #####
# ts.loc[prev_time,  'water_prices'] = 100 * 1000
#
# ##################### Value Function parameters #####################
#
# eta = 9000
# zd  = 105.69451612903225
# r   = 0.0041
#
# N = 10 # For the storage grid
# M = 10 # For the price grid
# W = 10 # For the water price grid
# L = 10 # For the outflow grid
# S = 12 # For the states
#
# max_ite = 60
# ite     = 0
# tol     = 2e-15
#
# discount = 1/(1+r)
#
# P_min_e = 6.5 / 100
# P_max_e = 11  / 100
#
# # P_min_e = 6.5
# # P_max_e = 11
#
# P_min_w = 60000
# P_max_w = 200000
#
# I_min = 1750
# I_max = 3500
# boundaries = np.array([2000,3000])
#
# q_min = 0
# q_max = 245
#
# rho = 2.33
# K = 10 # number of shocks in the storage grid
# J = 10 # number of shocks in the prices grid
#
# #################################
#
# # correlation between residuals:
#
# storage_res = np.loadtxt('Models/coeficients/residuals/inventory_residuals.csv')[1:]
# prices_res  = np.loadtxt('Models/coeficients/residuals/price_e_residuals.csv')
#
# correlation = np.corrcoef(storage_res,prices_res)[0,1]
#
# discount = 1/(1+r)
#
# elev_stor = zip(ts.elevation.values,ts.storage.values)
# elev_stor = sorted(elev_stor, key=lambda x: (x[1],x[0]))
#
# price_grid_e = get_middle_points(np.arange(P_min_e, P_max_e+ 2**-15, (P_max_e - P_min_e)/M))
# price_grid_w = get_middle_points(np.arange(P_min_w, P_max_w+ 2**-15, (P_max_w - P_min_w)/W))
# storage_grid = get_middle_points(np.arange(I_min, I_max+ 2**-15, (I_max - I_min)/N))
#
# shocks_storage  = get_middle_points(np.arange(-rho, rho+2**-15, (2*rho)/ K))
#
# dist_shocks_storage = (shocks_storage[1] - shocks_storage[0])/2
# prob_shocks_storage = norm.cdf(shocks_storage[:] + dist_shocks_storage) - norm.cdf(shocks_storage[:] - dist_shocks_storage)
#
# shocks_prices_e  = get_middle_points(np.arange(-rho, rho+2**-15, (2*rho)/ J))
#
# dist_shocks_prices_e = (shocks_prices_e[1] - shocks_prices_e[0])/2
# prob_shocks_prices_e = norm.cdf(shocks_prices_e[:] + dist_shocks_prices_e) - norm.cdf(shocks_prices_e[:] - dist_shocks_prices_e)
#
#
# shocks_prices_w = get_middle_points(np.arange(-rho, rho+2**-15, (2*rho)/ L))
#
# dist_shocks_prices_w = (shocks_prices_w[1] - shocks_prices_w[0])/2
# prob_shocks_prices_w = norm.cdf(shocks_prices_w[:] + dist_shocks_prices_w) - norm.cdf(shocks_prices_w[:] - dist_shocks_prices_w)
#
# A,B,C = np.ix_(prob_shocks_storage,
#                prob_shocks_prices_e,
#                prob_shocks_prices_w)
#
# A = A*B*C
#
# del B,C
#
# outflow_grid = np.arange(q_min, q_max+ 2**-15, (q_max - q_min)/L) # L+1 elements
#
# V0 = np.zeros((S,N,M,W))
# V1 = np.zeros((S,N,M,W))
#
# policy = np.zeros((S,N,M,W))
#
# cashflow_e = np.zeros((S,N,M,W))
# cashflow_w = np.zeros((S,N,M,W))
#
# step_price_e = price_grid_e[1] - price_grid_e[0]
# step_price_w = price_grid_w[1] - price_grid_w[0]
# step_storage = storage_grid[1] - storage_grid[0]

time_0 = 0 ## Because the last month is december int(str(ts.index.values[-1]).split('-')[1]) % 12
t0 = time()
while ite < max_ite:

    policy, V1, cashflow_e, cashflow_w = iv_algo(policy,
                                                 V0,
                                                 V1,
                                                 time_0,
                                                 cashflow_e,
                                                 cashflow_w,
                                                 S,
                                                 N,
                                                 M,
                                                 W,
                                                 storage_grid,
                                                 price_grid_e,
                                                 price_grid_w,
                                                 L,
                                                 outflow_grid,
                                                 A,
                                                 shocks_storage,
                                                 shocks_prices_e,
                                                 shocks_prices_w,
                                                 boundaries,
                                                 step_storage,
                                                 step_price_e,
                                                 step_price_w,
                                                 K,
                                                 J,
                                                 discount,
                                                 eta,
                                                 elev_stor,
                                                 zd)

    error = np.max(np.abs(V0-V1))/np.max(V1)
    if error < tol:
        V0 = V1.copy()
        print(time()-t0)
        print(f'Iteration: {ite}')
        print(f'Error is: {error}')
        break

    V0   = V1.copy()
    ite += 1
    print(time()-t0)
    print(f'Iteration: {ite}')
    print(f'Error is: {error}')
    print('='*50)


save_load_policy_V0(S,N,M,W, policy=policy, V0= V0, cashflow_e = cashflow_e, cashflow_w = cashflow_w, save_results = True)
