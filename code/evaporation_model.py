import pandas as pd
import numpy  as np
from nls_optimizer import *
from data_tools import load_data

ts = load_data()

def model_evaporation_0(coefs, t):
    return coefs[0] + coefs[1] * np.sin((coefs[2]+t)*2*np.pi/12)

evaporation = ts.evaporation.values
times       = np.arange(0,len(ts.index),1) % 12

def residuals_evaporation(coefs):
    return model_evaporation_0(coefs, times) - evaporation

num_coefs = 3
p0 = initial_values(num_coefs)
res_evaporation = NLS(residuals_evaporation, p0, xdata=times, ydata=evaporation)

print(res_evaporation.summary())
print(times[-1],ts.index[-1])


coefficients_evaporation = res_evaporation.parmEsts

np.savetxt('Models/coeficients/evaporation.csv', coefficients_evaporation, delimiter=',')

plt.plot(model_evaporation_0(coefficients_evaporation,times))
plt.plot(evaporation)
plt.show()
