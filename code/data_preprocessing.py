    import pandas as pd
import numpy  as np
from os import listdir
from os.path import isfile, join
import re
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import data_tools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import least_squares
from scipy import stats
import statsmodels.api as sm
import calendar
from my_time_series import (
    fit_AR_LS,
    residuals_AR,
    tests_gaussian_white_noise,
)

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

import data_tools
import itertools

from matplotlib import cm
from scipy.stats import norm
from Models.models import *
from time import time
from mpl_toolkits.mplot3d import axes3d, Axes3D
from datetime import datetime

from scipy.optimize import leastsq
import scipy.stats as spst
from numba import jit
from tabulate import tabulate
import scipy.fft
import math

import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter


dam_id = 1112

path = data_tools.from_id_to_time_series(dam_id)

dam_ts = pd.read_csv(path)
data_tools.add_year_month_database(dam_ts)

dam_ts = dam_ts.groupby('yearmonth').mean()

dam_ts.index = dam_ts.index.astype(int)

ele_ts = pd.read_csv('../APUS37A72610.csv')

## Convert string to numeric and interpolate missings (2 missings in electricity prices)

for idx in range(len(ele_ts.APUS37A72610)):

    elem = ele_ts.APUS37A72610[idx]
    if elem == '.':
        a = ele_ts.APUS37A72610[idx-1]
        b = eval(ele_ts.APUS37A72610[idx+1])

        ele_ts.APUS37A72610[idx] = (a+b)/2

    else:
        ele_ts.APUS37A72610[idx] = eval(elem)

## Convert DATE to yearmonth

for idx in ele_ts.index:
        date = ele_ts.loc[idx,'DATE']
        date = date.split('-')

        if np.isnan(ele_ts.APUS37A72610[idx]):
            continue

        ele_ts.loc[idx,'yearmonth'] = eval(date[0] + date[1])

ele_ts = ele_ts.drop('DATE',axis=1)
ele_ts = ele_ts.set_index('yearmonth')
ele_ts.index = ele_ts.index.astype(int)

## set same time intervales for both series

new_index = dam_ts.index.intersection(ele_ts.index)

dam_ts = dam_ts.loc[new_index,:]
ele_ts = ele_ts.loc[new_index,:]

ts = ele_ts.merge(dam_ts, left_index=True, right_index=True)
ts = ts.rename(columns= {'APUS37A72610': 'price'})

yearmonth = list(ts.index)
x_label   = data_tools.from_index_to_dates(yearmonth)

ts = ts.iloc[301:,:]
x_label = x_label[301:]

prices_data = pd.read_csv('../prices_processed.csv', index_col=0)
start = np.where(prices_data.index == 'Dec 2003')[0][0]
end   = np.where(prices_data.index == 'Jan 2020')[0][0]
new_prices = prices_data.iloc[start:end, -1].values

ts = ts.rename(columns={'price': 'price_power'})
ts.loc[:,'price_power'] = new_prices
ts = ts.set_index(np.array(list(map(lambda x: datetime( year = x // 100, month = x % 100, day = 15),ts.index.values))))

ts.to_csv('../Preprocessed_data/ts.csv')
