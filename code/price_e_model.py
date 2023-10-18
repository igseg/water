import pandas as pd
import numpy  as np
from nls_optimizer import *
from data_tools import load_data

ts = load_data()

prices = ts.price_power.values / 100
times  = np.arange(1,len(prices),1)

def model_2_rever(coefs,t):
    return prices[t-1] + coefs[0] +  coefs[1] * prices[t-1] + coefs[2] * (np.sin((t+coefs[3])*2*np.pi/6)) + coefs[4] * (np.sin((t+coefs[5])*2*np.pi/12))

def residuals_prices_rever_2(coefs):
    return model_2_rever(coefs, times[:]) - prices[1:]


num_coefs = 6
p0 = initial_values(num_coefs)
results_correlated_residuals_2 = NLS(residuals_prices_rever_2, p0, xdata=times, ydata=prices[1:])

print(results_correlated_residuals_2.summary())

np.savetxt('Models/coeficients/prices_e.csv', results_correlated_residuals_2.parmEsts, delimiter=',')
np.savetxt('Models/coeficients/prices_e_sd.csv', np.array([results_correlated_residuals_2.RMSE]), delimiter=',')
