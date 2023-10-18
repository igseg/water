import pandas as pd
import numpy  as np
from nls_optimizer import *
import statsmodels.api as sm
from data_tools import load_data

def year_transform(x):
    if x < 50:
        return x + 2000
    else:
        return x + 1900

index = pd.read_csv('../data_index_water_companies.csv', index_col = 0)

index['Date'] = (list(map(lambda x: datetime( year = year_transform(int(x.split('/')[2])) , month = int(x.split('/')[0]) , day = 15) , index.index.values)))

index = index.set_index('Date')

index = index.groupby(pd.Grouper(freq='M')).mean()
index['Date'] = (list(map(lambda x: datetime( year = x.year , month = x.month , day = 15) ,pd.DatetimeIndex(index.index.values))))
index = index.set_index('Date')
index = index.rename(columns={'Last Price': 'water_index'})

SP500 = pd.read_csv('../data_sp.csv', index_col=0)
SP500['Date'] = (list(map(lambda x: datetime( year = int(x.split('/')[2]) , month = int(x.split('/')[0]) , day = 15) ,SP500.index.values)))
SP500 = SP500.set_index('Date')
SP500 = SP500.drop( ['Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1)
SP500['Price'] = SP500.Price.apply(lambda x: x.replace(",", ""))
SP500['Price'] = SP500['Price'].astype('float')

SP500 = SP500.groupby(pd.Grouper(freq='M')).mean()
SP500['Date'] = (list(map(lambda x: datetime( year = x.year , month = x.month , day = 15) ,pd.DatetimeIndex(SP500.index.values))))
SP500 = SP500.set_index('Date')

data = index.merge(SP500, left_index=True, right_index=True)

y = np.diff(np.log(data['water_index'])) - 0.02 / 12
x = np.diff(np.log(data['Price'])) - 0.02 / 12
x = x.transpose()
x = sm.add_constant(x)
result = sm.OLS(y, x).fit()

beta = result.params[1]

expected_monthly_drift = 0.02 / 12 + (np.diff(np.log(data['Price'])).mean() - 0.02 / 12) * beta
std = result.resid.std() * beta

params = np.array([expected_monthly_drift, std])
np.savetxt('Models/coeficients/CAPM_water.csv', params, delimiter=',')

print(params)
