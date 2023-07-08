
from numpy import loadtxt, sin, pi,log, exp, min, max

coefs_evaporation = loadtxt('Models/coeficients/evaporation.csv')

def model_evaporation(t):
    return coefs_evaporation[0] + coefs_evaporation[1] * sin((coefs_evaporation[2]+t)*2*pi/12)

coefs_inflow = loadtxt('Models/coeficients/inflow_coef.csv')

def model_inflow(prev, t):
    return coefs_inflow[0] + coefs_inflow[1]* prev + coefs_inflow[2] * (sin((t+coefs_inflow[3])*2*pi/12))

def model_storage(model_inflow, prev_inflow , model_evaporation, t, prev_storage, outflow):
    inflow      = model_inflow(prev_inflow, t)
    evaporation = model_evaporation(t)

    return prev_storage + 2.592 * inflow - evaporation -  2.592 * outflow

coefs_log_prices = loadtxt('Models/coeficients/log_prices.csv')
coefs_mean_rever = loadtxt('Models/coeficients/mean_rever.csv')

coefs_log_prices_nonmr  = loadtxt('Models/coeficients/log_diff_prices_nonmr.csv')
coefs_log_prices_mr     = loadtxt('Models/coeficients/log_diff_prices_mr.csv'   )
coefs_prices_corr_resid = loadtxt('Models/coeficients/correlated_residuals.csv' )

######################### Relevant coefs for prices ######################

coefs_prices_cte   = loadtxt('Models/coeficients/prices_cte.csv')
coefs_prices_rever = loadtxt('Models/coeficients/prices_rever.csv')

coefs_prices_cte_2   = loadtxt('Models/coeficients/prices_2_cte.csv')
coefs_prices_rever_2 = loadtxt('Models/coeficients/prices_2_rever.csv')

coefs_water_prices = loadtxt('Models/coeficients/prices_water_v0.csv')


def model_prices_cte(t, prev):
    return prev + coefs_prices_cte[0] +  coefs_prices_cte[1] * (sin((t+coefs_prices_cte[2])*2*pi/6))

def model_prices_rever(t,prev):
    return prev + coefs_prices_rever[0] +  coefs_prices_rever[1] * prev + coefs_prices_rever[2] * (sin((t+coefs_prices_rever[3])*2*pi/6))

def model_water_prices(prev):
    return coefs_water_prices[0] * (coefs_water_prices[1] - prev)

# def model_2_cte(t,prev):
#     return prices[t-1] + coefs_prices_cte_2[0] +  coefs_prices_cte_2[1] * (np.sin((t+coefs_prices_cte_2[2])*2*np.pi/6)) + coefs_prices_cte_2[3] * (np.sin((t+coefs_prices_cte_2[4])*2*np.pi/12))
#
def model_2_rever(t,prev):
    return prev + coefs_prices_rever_2[0] +  coefs_prices_rever_2[1] * prev + coefs_prices_rever_2[2] * (sin((t+coefs_prices_rever_2[3])*2* pi/6)) + coefs_prices_rever_2[4] * (sin((t+coefs_prices_rever_2[5])*2*pi/12))


## This has to be redifined properly. It is used to compute the correlation between the random processes
def mean_rever(prev_x,prev_y_x):
    return coefs_mean_rever[0] * prev_x + coefs_mean_rever[1] * prev_y_x

def residual(prev_price, curr_price,t):
    f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
    return log(curr_price) - log(prev_price) - f(t)

def residual_with_X(prev_price,curr_price,X,t):
    f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
    return log(curr_price) - log(prev_price) - f(t) - X
# end noise

storage_sd = 103.79512935238394
prices_sd  = 0.027060425667826772
prices_sd_nonmr = 0.030060183797506825 # This is the first model of the fishery paper UPDATE
prices_sd_mr    = 0.029386989679649242 # This is the second model of the fishery paper UPDTE

prices_cte_sd   = 0.23525918886981498
prices_rever_sd = 0.23183635480506956

prices_2_cte_sd   = 0.0020533845309708318
prices_2_rever_sd = 0.0020295022531839801

water_prices_sd = 0.1766

def next_storage(model_inflow, prev_inflow , model_evaporation, t, prev_storage, outflow, model_storage, shock_n, scale = 1):
    return model_storage(model_inflow, prev_inflow , model_evaporation, t, prev_storage, outflow) + scale * storage_sd * shock_n


def next_price_cte(t, prev, shock_m):
    return model_prices_cte(t, prev)  + prices_cte_sd * shock_m

def next_price_rever(t, prev, shock_m):
    return model_prices_rever(t, prev)  + prices_rever_sd * shock_m

def next_water_price(prev, shock_w):
    return model_water_prices(prev) + water_prices_sd * shock_w

# def next_price(t,prev, shock_m):
#     f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
#     return prev*exp(f(t) + prices_sd * shock_m)
#
# def next_price_with_X(t,prev, prev_x, prev_y_x, shock_m):
#     f = lambda x: coefs_log_prices[0] + coefs_log_prices[1] * sin((coefs_log_prices[2]+x)*2*pi/12)
#     X = lambda x,x_y: coefs_mean_rever[0]*x + coefs_mean_rever[1] * x_y
#     return prev*exp(f(t) + X(prev_x, prev_y_x) + prices_sd * shock_m)

def z(I,elev_stor):
    unzipped = list(zip(*elev_stor))

    if I <= min(unzipped[1]):
        return elev_stor[0][0]

    if I >= max(unzipped[1]):
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

# coefs   = loadtxt('Models/coeficients/inflow_coef.csv')
# inflows = loadtxt('Models/coeficients/inflows.csv')

# def model_inflow(t):
#     return coefs[0]*inflows[t-1]+ coefs[1]*(1-coefs[0]) + (coefs[2] * (t% 12 == 1) + coefs[3] * (t% 12 == 2) + coefs[4] * (t% 12 == 3) + coefs[5] * (t% 12 == 4) + coefs[6] * (t% 12 == 5) + coefs[7] * (t% 12 == 6) + coefs[8] * (t% 12 == 7) + coefs[9] * (t% 12 == 8) + coefs[10] * (t% 12 == 9) + coefs[11] * (t% 12 == 10)+ coefs[12] * (t% 12 == 11))*(1- coefs[0])
