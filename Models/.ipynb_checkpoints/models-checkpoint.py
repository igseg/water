
from numpy import loadtxt, sin, pi

coefs = loadtxt('Models/coeficients/evaporation.csv')

def model_evaporation(t):
    return coefs[0] + coefs[1] * sin((coefs[2]+t)*2*pi/12)
