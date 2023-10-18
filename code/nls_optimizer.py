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

import statsmodels.api as sm

import data_tools
import itertools

from matplotlib import cm
from time import time
from mpl_toolkits.mplot3d import axes3d, Axes3D
from datetime import datetime

from scipy.optimize import leastsq
import scipy.stats as spst
from numba import jit
from tabulate import tabulate
import scipy.fft
import math


def initial_values(num_coefs):
    p0 = {}
    for i in range(num_coefs):
        p0[f'coefs{i}'] = 0

    return p0

class NLS:
    ''' This provides a wrapper for scipy.optimize.leastsq to get the relevant output for nonlinear least squares.
    Although scipy provides curve_fit for that reason, curve_fit only returns parameter estimates and covariances.
    This wrapper returns numerous statistics and diagnostics'''


    def __init__(self, func, p0, xdata, ydata):
        # Check the data
        if len(xdata) != len(ydata):
            msg = 'The number of observations does not match the number of rows for the predictors'
            raise ValueError(msg)

        # Check parameter estimates
        if type(p0) != dict:
            msg = "Initial parameter estimates (p0) must be a dictionry of form p0={'a':1, 'b':2, etc}"
            raise ValueError(msg)

        self.func  = func
        self.inits = list(p0.values())
        self.xdata = xdata
        self.ydata = ydata
        self.nobs  = len( ydata )
        self.nparm = len( self.inits )

        self.parmNames = p0.keys()

        # Run the model
        self.mod1 = leastsq(self.func, self.inits, full_output=1)

        # Get the parameters
        self.parmEsts = np.round( self.mod1[0], 4)

        # Get the Error variance and standard deviation
        self.RSS  = np.sum( self.mod1[2]['fvec']**2 ) # RSS
        self.df   = self.nobs - self.nparm            # degrees of freedom
        self.MSE  = self.RSS / self.df                # mean squared error
        self.RMSE = np.sqrt( self.MSE )               # root mean squared error

        # Get the covariance matrix
        self.cov = self.MSE * self.mod1[1] # it is not clear what mod1[1] is

        # Get parameter standard errors
        self.parmSE = np.sqrt( np.diag( self.cov ) )

        # Calculate the t-values
        self.tvals = self.parmEsts/self.parmSE

        # Get p-values
        self.pvals = (1 - spst.t.cdf( np.abs(self.tvals), self.df))*2

        # Get biased variance (MLE) and calculate log-likehood
        self.s2b = self.RSS / self.nobs
        self.logLik = -self.nobs/2 * np.log(2*np.pi) - self.nobs/2 * np.log(self.s2b) - 1/(2*self.s2b) * self.RSS

        del(self.mod1)
        del(self.s2b)
        del(self.inits)

    # Get AIC. Add 1 to the df to account for estimation of standard error
    def AIC(self, k=2):
        return -2*self.logLik + k*(self.nparm + 1)

    # Print the summary
    def summary(self):
        print('Non-linear least squares')
        print('Model: ' + self.func.__name__)
        table = [['Variable', 'Estimate', 'Std. Error', 't-value', 'P(>|t|)']]
        for i in range( len(self.parmNames) ):
                row = [f'{list(self.parmNames)[i]}', f'{self.parmEsts[i]:5.4f}', f'{self.parmSE[i]:5.4f}', f'{self.tvals[i]:5.4f}', f'{self.pvals[i]:5.4f}']
                table.append(row)

        print(tabulate(table, tablefmt='fancy_grid'))
        print(f'Residual Standard Error: {self.RMSE: 5.4f}')
        print(f'Df: {int(self.df)}')
