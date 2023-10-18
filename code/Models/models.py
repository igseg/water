import numpy  as np
from numba import jit

inventory_sd      = float(np.loadtxt('Models/coeficients/inventory_sd.csv'))
coefs_evaporation = np.loadtxt('Models/coeficients/evaporation.csv')
coefs_inflow      = np.loadtxt('Models/coeficients/inflow_coef.csv')

@jit
def model_inflow_numba(t):
    return coefs_inflow[0] + coefs_inflow[1] * (np.sin((t+coefs_inflow[2])*2*np.pi/12))

@jit
def model_evaporation_numba(t):
    return coefs_evaporation[0] + coefs_evaporation[1] * np.sin((coefs_evaporation[2]+t)*2*np.pi/12)

@jit
def next_storage_numba(t, prev_storage, outflow, shock_n): ### ELIMINATE PREV_INFLOW
    inflow  = model_inflow_numba(t)
    evapor  = model_evaporation_numba(t)
    storage = prev_storage + 2.592 * inflow - evapor -  2.592 * outflow
    return storage + inventory_sd * shock_n

coefs_prices_e = np.loadtxt('Models/coeficients/prices_e.csv')
prices_e_sd = float(np.loadtxt('Models/coeficients/prices_e_sd.csv'))

@jit
def model_price_e(t,prev):
    return prev + coefs_prices_e[0] +  coefs_prices_e[1] * prev + coefs_prices_e[2] * (np.sin((t+coefs_prices_e[3])*2*np.pi/6)) + coefs_prices_e[4] * (np.sin((t+coefs_prices_e[5])*2*np.pi/12))

@jit
def next_price_e(t, prev, shock_m):
    return model_price_e(t, prev)  + prices_e_sd * shock_m

coefs_water_prices = np.loadtxt('Models/coeficients/CAPM_water.csv')
drift_water = coefs_water_prices[0]
prices_w_sd = coefs_water_prices[1]

@jit
def model_water_prices(prev):
    return prev  * np.exp(drift_water)

@jit
def next_water_price(prev, shock_w):
    #return np.exp(np.log(model_water_prices(prev,t)) + water_prices_sd * shock_w)
    return model_water_prices(prev) * np.exp(prices_w_sd * shock_w)

def get_middle_points(x):
    x = np.array(x)
    return (x[:-1] + x[1:])/2

def save_load_policy_V0(S,N,M,W, policy = None, V0 = None, cashflow_e = None, cashflow_w = None,
                       save_results = False, load_results = False):

    if save_results and policy.any() and V0.any():
        policy.tofile('Results/policy_2.txt')
        V0.tofile('Results/V0_2.txt')
        cashflow_e.tofile('Results/cashflow_e_2.txt')
        cashflow_w.tofile('Results/cashflow_w_2.txt')

    if load_results:
        mr_file = open("Results/policy_2.txt")
        loaded_policy = np.fromfile(mr_file)
        loaded_policy = loaded_policy.reshape((S, N, M, W))
        mr_file.close()

        mr_file = open("Results/cashflow_e_2.txt")
        loaded_cashflow_e = np.fromfile(mr_file)
        loaded_cashflow_e = loaded_cashflow_e.reshape((S, N, M, W))
        mr_file.close()

        mr_file = open("Results/cashflow_w_2.txt")
        loaded_cashflow_w = np.fromfile(mr_file)
        loaded_cashflow_w = loaded_cashflow_w.reshape((S, N, M, W))
        mr_file.close()

        V0_mr = open("Results/V0_2.txt")
        loaded_V0 = np.fromfile(V0_mr)
        loaded_V0 = loaded_V0.reshape((S, N, M, W))
        V0_mr.close()
        return loaded_policy, loaded_V0, loaded_cashflow_e, loaded_cashflow_w
    return None

@jit
def z_numba(I,elev_stor):
    if I <= elev_stor[0][1]:
        return elev_stor[0][0]

    if I >= elev_stor[-1][1]:
        return elev_stor[-1][0]

    for idx in range(1,len(elev_stor)):
        prev_storage = elev_stor[idx-1][1]
        curr_storage = elev_stor[idx][1]

        if prev_storage == curr_storage:
            continue

        if I >= prev_storage and I < curr_storage:
            prev_height = elev_stor[idx-1][0]
            curr_height = elev_stor[idx][0]

            return prev_height   + (curr_height-prev_height) * (I - prev_storage)/(curr_storage-prev_storage)

@jit
def pi_fun_numba(storage, t, price_e, price_w, q, eta, elev_stor, zd):
    # inflow  = coefs_inflow[0] + coefs_inflow[2] * (np.sin((t+coefs_inflow[3])*2*np.pi/12))
    # evapor  = coefs_evaporation[0] + coefs_evaporation[1] * np.sin((coefs_evaporation[2]+t)*2*np.pi/12)
    # expected_storage = storage + 2.592 * inflow - evapor -  2.592 * q

    expected_storage = next_storage_numba(t, storage, q, 0)
    return eta * q * (1/2.592) * ((z_numba(storage, elev_stor) + z_numba(expected_storage,elev_stor))/2 - zd) * price_e  ,  price_w * q

@jit
def prepro_coord_array_numba(x,grid):

    cond = np.logical_or(x<grid[0], x>grid[-1])
    x[x<grid[0]]  = grid[0]
    x[x>grid[-1]] = grid[-1]

    return x, cond

@jit
def coord_array_numba(x, grid, step):
    return ((x - grid[0])/step + 0.5).astype(np.int_)

# @jit
# def coord_array_upper_numba(x,grid,x_cond,step):
#     return ((x - grid[0])/step + 1*(~x_cond)).astype(np.int_)
#
# @jit
# def coord_array_lower_numba(x,grid, step):
#     return ((x - grid[0])/step).astype(np.int_)
#
@jit
def gridding(future_storage,
             future_price_e,
             future_price_w,
             price_grid_e,
             price_grid_w,
             storage_grid,
             value,
             step_storage,
             step_price_e,
             step_price_w,
             K,
             J,
             L):


    future_storage, cond_storage = prepro_coord_array_numba(future_storage, storage_grid)
    future_price_e, cond_price_e = prepro_coord_array_numba(future_price_e, price_grid_e)
    future_price_w, cond_price_w = prepro_coord_array_numba(future_price_w, price_grid_w)

    ## the condition is true if the value is out of the boundary

#     future_storage_upper = coord_array_upper_numba(future_storage, storage_grid, cond_storage, step_storage)
#     future_price_e_upper = coord_array_upper_numba(future_price_e, price_grid_e, cond_price_e, step_price_e)
#     future_price_w_upper = coord_array_upper_numba(future_price_w, price_grid_w, cond_price_w, step_price_w)

#     future_storage_lower = coord_array_lower_numba(future_storage, storage_grid, step_storage)
#     future_price_e_lower = coord_array_lower_numba(future_price_e, price_grid_e, step_price_e)
#     future_price_w_lower = coord_array_lower_numba(future_price_w, price_grid_w, step_price_w)

    future_storage_closest = coord_array_numba(future_storage, storage_grid, step_storage)
    future_price_e_closest = coord_array_numba(future_price_e, price_grid_e, step_price_e)
    future_price_w_closest = coord_array_numba(future_price_w, price_grid_w, step_price_w)

    #print(future_price_closest)

#     future_storage_upper[cond_storage] = future_storage_closest[cond_storage]
#     future_price_e_upper[cond_price_e] = future_price_e_closest[cond_price_e]
#     future_price_w_upper[cond_price_w] = future_price_w_closest[cond_price_w]

#     future_storage_exact = coord_array_exact_numba(future_storage, storage_grid, step_storage)
#     future_price_e_exact = coord_array_exact_numba(future_price_e, price_grid_e, step_price_e)
#     future_price_w_exact = coord_array_exact_numba(future_price_w, price_grid_w, step_price_w)

    # everything is in coordinates, so i can just work that here.

    future_value = np.zeros((K,J,L))

    for x in range(K):
        for y in range(J):
            for z in range(L):
                ##### Hay que meter las 8 sumas
                sol = 0
                sol = value[future_storage_closest[x], future_price_e_closest[y], future_price_w_closest[z]]

#                 sol += ((future_storage_upper[x] - future_storage_exact[x]) * (future_price_e_upper[y] - future_price_e_exact[y]) *
#                 (future_price_w_upper[z] - future_price_w_exact[z]) * value[future_storage_upper[x], future_price_e_upper[y], future_price_w_upper[z]]  )# [1,1,1]

#                 sol += ((future_storage_upper[x] - future_storage_exact[x]) * (future_price_e_upper[y] - future_price_e_exact[y]) *
#                 (future_price_w_lower[z] - future_price_w_exact[z]) * (-1) * value[future_storage_upper[x], future_price_e_upper[y], future_price_w_lower[z]] )# [1,1,0]

#                 sol += ((future_storage_upper[x] - future_storage_exact[x]) * (future_price_e_lower[y] - future_price_e_exact[y]) * (-1) *
#                 (future_price_w_upper[z] - future_price_w_exact[z]) * value[future_storage_upper[x], future_price_e_lower[y], future_price_w_upper[z]] )# [1,0,1]

#                 sol += ((future_storage_upper[x] - future_storage_exact[x]) * (future_price_e_lower[y] - future_price_e_exact[y]) * (-1) *
#                 (future_price_w_lower[z] - future_price_w_exact[z]) * (-1) * value[future_storage_upper[x], future_price_e_lower[y], future_price_w_lower[z]] )# [1,0,0]

#                 sol += ((future_storage_lower[x] - future_storage_exact[x]) * (-1) * (future_price_e_upper[y] - future_price_e_exact[y]) *
#                 (future_price_w_upper[z] - future_price_w_exact[z]) * value[future_storage_lower[x], future_price_e_upper[y], future_price_w_upper[z]] )# [0,1,1]

#                 sol += ((future_storage_lower[x] - future_storage_exact[x]) * (-1) * (future_price_e_upper[y] - future_price_e_exact[y]) *
#                 (future_price_w_lower[z] - future_price_w_exact[z]) * (-1) * value[future_storage_lower[x], future_price_e_upper[y], future_price_w_lower[z]] )# [0,1,0]

#                 sol += ((future_storage_lower[x] - future_storage_exact[x]) * (-1) * (future_price_e_lower[y] - future_price_e_exact[y]) * (-1) *
#                 (future_price_w_upper[z] - future_price_w_exact[z]) * value[future_storage_lower[x], future_price_e_lower[y], future_price_w_upper[z]])# [0,0,1]

#                 sol += ((future_storage_lower[x] - future_storage_exact[x]) * (-1) * (future_price_e_lower[y] - future_price_e_exact[y]) * (-1) *
#                 (future_price_w_lower[z] - future_price_w_exact[z]) * (-1) * value[future_storage_lower[x], future_price_e_lower[y], future_price_w_lower[z]] )# [0,0,0]


                future_value[x,y,z] = sol

    return future_value

@jit
def future_payments_numba(Value_next,
                          A,
                          shocks_storage,
                          shocks_prices_e,
                          shocks_prices_w,
                          # prev_inflow,
                          current_time,
                          q,
                          storage,
                          price_e,
                          price_w,
                          price_grid_e,
                          price_grid_w,
                          storage_grid,
                          boundaries,
                          step_storage,
                          step_price_e,
                          step_price_w,
                          K,
                          J,
                          L):

    #future_payment = np.zeros((K,J))

    future_storage = next_storage_numba(current_time + 1, storage, q, shocks_storage) # stays the same
    cond = np.logical_or(future_storage < boundaries[0], future_storage > boundaries[1])  # stays the same

    ###################

    future_price_e = next_price_e(current_time + 1, price_e  , shocks_prices_e)

    ###################
    # add water price

    future_price_w = next_water_price(price_w , shocks_prices_w) ## make this to numba prolly.


    ###################

    under_storage = False # stays the same
    over_storage  = False # stays the same

    if future_storage.min() < boundaries[0]: # stays the same
        under_storage = True # stays the same

    elif future_storage.max() > boundaries[1]: # stays the same
        over_storage  = True # stays the same

    # Calculate the value at each future state

    future_value = gridding(future_storage = future_storage,
                            future_price_e = future_price_e,
                            future_price_w = future_price_w,
                            price_grid_e   = price_grid_e,
                            price_grid_w   = price_grid_w,
                            storage_grid   = storage_grid,
                            value          = Value_next,
                            step_storage   = step_storage,
                            step_price_e   = step_price_e,
                            step_price_w   = step_price_w,
                            K              = K,
                            J              = J,
                            L              = L)

    future_payment = np.multiply(future_value, A) # Element wise # stays the same

    # Compute the expected value and the value is 0 if it violates boundary conditions

    sol = 0

    for x in range(cond.shape[0]):
        if not cond[x]:
            sol += np.sum(future_payment[x,:,:])

    return sol, under_storage, over_storage

@jit
def action_value_numba(state,
                        # prev_inflow,
                        current_time,
                        q,
                        storage,
                        price_e,
                        price_w,
                        V0,
                        A,
                        shocks_storage,
                        shocks_prices_e,
                        shocks_prices_w,
                        price_grid_e,
                        price_grid_w,
                        storage_grid,
                        boundaries,
                        step_storage,
                        step_price_e,
                        step_price_w,
                        K,
                        J,
                        L,
                        discount,
                        eta,
                        elev_stor,
                        zd):

    s_next = state+1
    next_state_value, under_storage, over_storage = future_payments_numba(Value_next      = V0[s_next % 12],
                                                                          A               = A,
                                                                          shocks_storage  = shocks_storage,
                                                                          shocks_prices_e = shocks_prices_e,
                                                                          shocks_prices_w = shocks_prices_w,
                                                                          # prev_inflow     = prev_inflow,
                                                                          current_time    = current_time,
                                                                          q               = q,
                                                                          storage         = storage,
                                                                          price_e         = price_e,
                                                                          price_w         = price_w,
                                                                          price_grid_e    = price_grid_e,
                                                                          price_grid_w    = price_grid_w,
                                                                          storage_grid    = storage_grid,
                                                                          boundaries      = boundaries,
                                                                          step_storage    = step_storage,
                                                                          step_price_e    = step_price_e,
                                                                          step_price_w    = step_price_w,
                                                                          K               = K,
                                                                          J               = J,
                                                                          L               = L)

    next_state_value = discount * next_state_value

    if under_storage or over_storage:
        return 0 + next_state_value, 0, 0
    else:
#         return pi_fun_numba(storage, prev_inflow, current_time, price_e, price_w, q) + next_state_value
        elec_cf, water_cf = pi_fun_numba(storage, current_time + 1, price_e, price_w, q, eta, elev_stor, zd)

        return  elec_cf + water_cf + next_state_value, elec_cf, water_cf

@jit
def iv_algo(policy,
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
            zd):
    for s in range(S):
        current_time = time_0 + s
        # prev_inflow  = inflow_values[current_time] ## inflow used to forecast the next step

        for n in range(N):
            storage = storage_grid[n]

            for m in range(M):
                price_e = price_grid_e[m]

                for w in range(W):
                    price_w = price_grid_w[w]

                    rewards = np.zeros(L+1)
                    elec_cf = np.zeros(L+1)
                    water_cf= np.zeros(L+1)

                    i = 0
                    #input_next_storage = [model_inflow, inflow , model_evaporation, current_time,
                    #          storage, 0, model_storage]


                    for q in outflow_grid:

                        rewards[i], elec_cf[i], water_cf[i] = (action_value_numba(state = s,
                                                         # prev_inflow     = prev_inflow,
                                                         current_time    = current_time,
                                                         q               = q,
                                                         storage         = storage,
                                                         price_e         = price_e,
                                                         price_w         = price_w,
                                                         V0              = V0,
                                                         A               = A,
                                                         shocks_storage  = shocks_storage,
                                                         shocks_prices_e = shocks_prices_e,
                                                         shocks_prices_w = shocks_prices_w,
                                                         price_grid_e    = price_grid_e,
                                                         price_grid_w    = price_grid_w,
                                                         storage_grid    = storage_grid,
                                                         boundaries      = boundaries,
                                                         step_storage    = step_storage,
                                                         step_price_e    = step_price_e,
                                                         step_price_w    = step_price_w,
                                                         K               = K,
                                                         J               = J,
                                                         L               = L,
                                                         discount        = discount,
                                                         eta             = eta,
                                                         elev_stor       = elev_stor,
                                                         zd              = zd))
                        i+=1

                    policy[s,n,m,w] = outflow_grid[np.argmax(rewards)]
                    V1[s,n,m,w]     = np.max(rewards)
                    cashflow_e[s,n,m,w] = elec_cf[np.argmax(rewards)]
                    cashflow_w[s,n,m,w] = water_cf[np.argmax(rewards)]

    return policy, V1, cashflow_e, cashflow_w








############################### LEGACY ###############################




# from numpy import loadtxt, sin, pi,log, exp, min, max
#
# coefs_evaporation = loadtxt('Models/coeficients/evaporation.csv')
#
# def model_evaporation(t):
#     return coefs_evaporation[0] + coefs_evaporation[1] * sin((coefs_evaporation[2]+t)*2*pi/12)
#
# coefs_inflow = loadtxt('Models/coeficients/inflow_coef.csv')
#
# def model_inflow(prev, t):
#     return coefs_inflow[0] + coefs_inflow[1]* prev + coefs_inflow[2] * (sin((t+coefs_inflow[3])*2*pi/12))
#
# def model_storage(model_inflow, prev_inflow , model_evaporation, t, prev_storage, outflow):
#     inflow      = model_inflow(prev_inflow, t)
#     evaporation = model_evaporation(t)
#
#     return prev_storage + 2.592 * inflow - evaporation -  2.592 * outflow
#
# coefs_log_prices = loadtxt('Models/coeficients/log_prices.csv')
# coefs_mean_rever = loadtxt('Models/coeficients/mean_rever.csv')
#
# coefs_log_prices_nonmr  = loadtxt('Models/coeficients/log_diff_prices_nonmr.csv')
# coefs_log_prices_mr     = loadtxt('Models/coeficients/log_diff_prices_mr.csv'   )
# coefs_prices_corr_resid = loadtxt('Models/coeficients/correlated_residuals.csv' )
#
# ######################### Relevant coefs for prices ######################
#
# coefs_prices_cte   = loadtxt('Models/coeficients/prices_cte.csv')
# coefs_prices_rever = loadtxt('Models/coeficients/prices_rever.csv')
#
# coefs_prices_cte_2   = loadtxt('Models/coeficients/prices_2_cte.csv')
# coefs_prices_rever_2 = loadtxt('Models/coeficients/prices_2_rever.csv')
#
# coefs_water_prices = loadtxt('Models/coeficients/prices_water_v0.csv')
#
#
# def model_prices_cte(t, prev):
#     return prev + coefs_prices_cte[0] +  coefs_prices_cte[1] * (sin((t+coefs_prices_cte[2])*2*pi/6))
#
# def model_prices_rever(t,prev):
#     return prev + coefs_prices_rever[0] +  coefs_prices_rever[1] * prev + coefs_prices_rever[2] * (sin((t+coefs_prices_rever[3])*2*pi/6))
#
# def model_water_prices(prev):
#     #return coefs_water_prices[0] * (coefs_water_prices[1] - prev)
#     return prev  * exp(0.003459281020099942)
#
# # def model_2_cte(t,prev):
# #     return prices[t-1] + coefs_prices_cte_2[0] +  coefs_prices_cte_2[1] * (np.sin((t+coefs_prices_cte_2[2])*2*np.pi/6)) + coefs_prices_cte_2[3] * (np.sin((t+coefs_prices_cte_2[4])*2*np.pi/12))
# #
# def model_2_rever(t,prev):
#     return prev + coefs_prices_rever_2[0] +  coefs_prices_rever_2[1] * prev + coefs_prices_rever_2[2] * (sin((t+coefs_prices_rever_2[3])*2* pi/6)) + coefs_prices_rever_2[4] * (sin((t+coefs_prices_rever_2[5])*2*pi/12))
#
#
# ## This has to be redifined properly. It is used to compute the correlation between the random processes
# def mean_rever(prev_x,prev_y_x):
#     return coefs_mean_rever[0] * prev_x + coefs_mean_rever[1] * prev_y_x
#
# def residual(prev_price, curr_price,t):
#     f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
#     return log(curr_price) - log(prev_price) - f(t)
#
# def residual_with_X(prev_price,curr_price,X,t):
#     f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
#     return log(curr_price) - log(prev_price) - f(t) - X
# # end noise
#
# storage_sd = 103.79512935238394
# prices_sd  = 0.027060425667826772
# prices_sd_nonmr = 0.030060183797506825 # This is the first model of the fishery paper UPDATE
# prices_sd_mr    = 0.029386989679649242 # This is the second model of the fishery paper UPDTE
#
# prices_cte_sd   = 0.23525918886981498
# prices_rever_sd = 0.23183635480506956
#
# prices_2_cte_sd   = 0.0020533845309708318
# prices_2_rever_sd = 0.0020295022531839801
#
# water_prices_sd = 0.1766
#
# def next_storage(model_inflow, prev_inflow , model_evaporation, t, prev_storage, outflow, model_storage, shock_n, scale = 1):
#     return model_storage(model_inflow, prev_inflow , model_evaporation, t, prev_storage, outflow) + scale * storage_sd * shock_n
#
#
# def next_price_cte(t, prev, shock_m):
#     return model_prices_cte(t, prev)  + prices_cte_sd * shock_m
#
# def next_price_rever(t, prev, shock_m):
#     return model_prices_rever(t, prev)  + prices_rever_sd * shock_m
#
# def next_water_price(prev, shock_w):
#     #return model_water_prices(prev) + water_prices_sd * shock_w
#     return model_water_prices(prev) * exp(0.035014571162931295 * shock_w)
#
# # def next_price(t,prev, shock_m):
# #     f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
# #     return prev*exp(f(t) + prices_sd * shock_m)
# #
# # def next_price_with_X(t,prev, prev_x, prev_y_x, shock_m):
# #     f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
# #     X = lambda x,x_y: coefs_mean_rever[0]*x + coefs_mean_rever[1] * x_y
# #     return prev*exp(f(t) + X(prev_x, prev_y_x) + prices_sd * shock_m)
#
# def z(I,elev_stor):
#     unzipped = list(zip(*elev_stor))
#
#     if I <= min(unzipped[1]):
#         return elev_stor[0][0]
#
#     if I >= max(unzipped[1]):
#         return elev_stor[-1][0]
#
#     for idx in range(1,len(elev_stor)):
#         prev_storage = elev_stor[idx-1][1]
#         curr_storage = elev_stor[idx][1]
#
#         if prev_storage == curr_storage:
#             continue
#
#         if I >= prev_storage and I < curr_storage:
#             prev_height = elev_stor[idx-1][0]
#             curr_height = elev_stor[idx][0]
#
#             return prev_height   + (curr_height-prev_height) * (I - prev_storage)/(curr_storage-prev_storage)
#
# # coefs   = loadtxt('Models/coeficients/inflow_coef.csv')
# # inflows = loadtxt('Models/coeficients/inflows.csv')
#
# # def model_inflow(t):
# #     return coefs[0]*inflows[t-1]+ coefs[1]*(1-coefs[0]) + (coefs[2] * (t% 12 == 1) + coefs[3] * (t% 12 == 2) + coefs[4] * (t% 12 == 3) + coefs[5] * (t% 12 == 4) + coefs[6] * (t% 12 == 5) + coefs[7] * (t% 12 == 6) + coefs[8] * (t% 12 == 7) + coefs[9] * (t% 12 == 8) + coefs[10] * (t% 12 == 9) + coefs[11] * (t% 12 == 10)+ coefs[12] * (t% 12 == 11))*(1- coefs[0])
