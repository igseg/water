import pandas as pd
import numpy  as np
from nls_optimizer import *
from data_tools import load_data

ts = load_data()

## Define the model.

inflows = ts.inflow.values
times   = np.arange(1,len(ts.index),1)

def model_inflow_0(coefs,t):
    # return coefs[0] + coefs[1]* inflows[t-1] + coefs[2] * (np.sin((t+coefs[3])*2*np.pi/12))
    return coefs[0] + coefs[1] * (np.sin((t+coefs[2])*2*np.pi/12))

def residuals_inflow(coefs):
    return model_inflow_0(coefs, times) - inflows[1:]

num_coefs = 3
p0 = initial_values(num_coefs)
inflow_res = NLS(residuals_inflow, p0, xdata=times, ydata=inflows[1:])

print(inflow_res.summary())

coefficients_inflow = inflow_res.parmEsts

np.savetxt('Models/coeficients/inflow_coef.csv', coefficients_inflow, delimiter=',')
# np.savetxt('Models/coeficients/inflows.csv', inflows, delimiter=',')

plt.plot(inflows, label = 'observed')
plt.plot(model_inflow_0(coefficients_inflow,times), label = 'prediction')
plt.legend()
plt.show()
