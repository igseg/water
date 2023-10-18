import pandas as pd
import numpy  as np
from nls_optimizer import *
from data_tools import load_data

ts = load_data()

coefficients_inflow      = np.loadtxt('Models/coeficients/inflow_coef.csv')
coefficients_evaporation = np.loadtxt('Models/coeficients/evaporation.csv')
inflows = ts.inflow.values

times   = np.arange(0,len(ts.index),1)

def model_evaporation_0(coefs, t):
    return coefs[0] + coefs[1] * np.sin((coefs[2]+t)*2*np.pi/12)

def model_inflow_0(coefs,t):
    # return coefs[0] + coefs[1]* inflows[t-1] + coefs[2] * (np.sin((t+coefs[3])*2*np.pi/12))
    return coefs[0] + coefs[1] * (np.sin((t+coefs[2])*2*np.pi/12))

def model_storage_0(t):
    inflow      = model_inflow_0(coefficients_inflow, t)
    evaporation = model_evaporation_0(coefficients_evaporation,t)

    return ts.storage.values[t-1] + 2.592 * inflow - evaporation -  2.592 * ts.outflow.values[t-1]

prediction = model_storage_0(times[1:])

plt.plot(times[1:], ts.storage[1:],label = 'Observed')
plt.plot(times[1:], prediction[:],label = 'Predicted')
plt.locator_params(axis='x', nbins=8)
plt.xlim(-5,194)
ax = plt.gca()
x = ax.get_xticks
l = []
for e in x()[:]:
    year, month = divmod(int(e) + 1, 12)
    year  += 4
    if year < 10:
        year = f'0{year}'
    month += 1
    month = calendar.month_name[month][:3]
    l.append(f'{month}-{year}')
ax.set_xticklabels(l)
plt.grid(axis='y')
plt.title('Inventory dynamic', fontdict={'size': 18})
plt.xlabel('Time (months)',fontdict={'size':15})
plt.ylabel('Reservoir Inventory (MCM)',fontdict={'size':15})
plt.legend(fontsize = 13)
plt.savefig('../Figures/paper/Inventory_dynamic.pdf', format = 'pdf')
plt.show()

# Residuals

residuals = ts.storage[1:] - prediction
np.savetxt('Models/coeficients/inventory_sd.csv', np.array([np.std(residuals)]), delimiter=',')
np.savetxt('Models/coeficients/residuals/inventory_residuals.csv', residuals, delimiter=',')

print(f'Storage residuals std: {np.std(residuals)}')
