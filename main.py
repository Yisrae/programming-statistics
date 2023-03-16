#!/usr/bin/python3

from math import factorial, sqrt
from statistics import *
from itertools import permutations, combinations, combinations_with_replacement
import pandas as pd
import numpy as np

# Return the sample arithmetic mean of data which can be a sequence or iterable.
def calcMean(params):
    return mean(params)

# Return the population standard deviation (the square root of the population variance).
def calcPstDev(params):
    return pstdev(args)

# Return the population variance of data, a non-empty sequence or iterable of real-valued numbers.
# Mean is optional
def calcPVariance(params, mean=None):
    return pvariance(params, mean)

# Return the sample standard deviation (the square root of the sample variance)
def calcStDev(params):
    return stdev(args)

# Return the sample variance of data, an iterable of at least two real-valued numbers. Variance, or second moment about the mean, is a measure of the variability (spread or dispersion) of data.
# Note: This is the sample variance s² with Bessel’s correction, also known as variance with N-1 degrees of freedom. Provided that the data
def calcVariance(params):
    return variance(params)

# Returns a new NormalDist object where mu represents the arithmetic mean and sigma represents the standard deviation.
def calcNormalDist(mu=0.0, sigma=1.0):
    return NormalDist(mu, sigma)

def calcNormalDistFromSamples(params):
    return NormalDist.from_samples(params)

# Return successive r length permutations of elements in the iterable.
def calcPermutations(iterable, r=None):
    perm = permutations(iterable, r)
    for i in list(perm):
        print(i)

# Return permutation
def nPr(n, r):
    return int(factorial(n)/factorial(n-r))

# Return combination
def nCr(n, r):
    return int(factorial(n)/(factorial(n-r)* factorial(r)))

# Return square of each list element
def _squareArr(list):
    return [i ** 2 for i in list]

# Example usage
args = [18, 22, 13, 15, 24, 24, 20, 19, 19, 12, 16, 25, 14, 19, 21, 23, 25, 18, 18, 13, 26, 26, 25, 25, 19, 17, 18, 15, 13, 21, 19, 19, 14, 24, 20, 21, 23, 22, 19, 17]
mean = calcMean(args)
sigma = 2.5
n = 6
r = 3

print('Arithmetic mean:', mean)
print('Population standard deviation:', calcPstDev(args))
print('Population variance:', calcPVariance(args, mean))
print('Sample standard deviation:', calcStDev(args))
print('Sample variance:', calcVariance(args))
print('Normal distribution:', calcNormalDist(mean, sigma))
print('Normal distribution from samples:', calcNormalDistFromSamples(args))
print('nPr:', nPr(n, r))
print('nCr:', nCr(n, r))

print('\nPROBABILITY DISTRIBUTION')

randomVariableX = [0,1,2,3]
probX = [0.46,0.41,0.09,0.04]

xProbX = [a * b for a, b in zip(randomVariableX, probX)]

d = {
     'x': randomVariableX,
     'P(x)': probX,
     'xP(x)': xProbX
 }

df = pd.DataFrame(data=d)
print(df)

print('\nTotal P(x): ', np.sum(probX))
xProbXMean = np.sum(xProbX)
print('Total xP(x) / Mean: ', xProbXMean)

sqrRandVarXArr = _squareArr(randomVariableX)

sqrProbX = [a * b for a, b in zip(sqrRandVarXArr, probX)]

sumSqrProbX = np.sum(sqrProbX)

std_dvt = sumSqrProbX - (xProbXMean ** 2)
standard_deviation = sqrt(std_dvt)

print('Standard Deviation (prod. dist.): ', standard_deviation)


# Only works for exact values
print('\nBINOMIAL DISTRIBUTION')

nVal = 6
rVal = 3 # same as xVal
pVal = 0.3
qVal = 1 - pVal

binProb = nCr(nVal, rVal) * (pVal ** rVal) * (qVal ** (nVal - rVal))
stdBinProb = sqrt(nVal * pVal * qVal)
meanBinProb = nVal * pVal

print('Probability (bin. dist.): ', binProb)
print('Standard Deviation (bin. dist.): ', stdBinProb)
print('Mean (bin. dist.): ', meanBinProb)


# Only works for exact values
print('\nPOISSON DISTRIBUTION')
nVal = 14000
pVal = 0.00003
muVal = nVal * pVal
xVal = 1
eVal = 2.71828 # Approximately

if (nVal >= 100) == False:
    exit('Fail n > 100 check')

if (muVal <= 10) == False:
    exit('Fail mean < 10 check')

poissonProb = ((muVal ** xVal) * (eVal ** -muVal)) / factorial(xVal)

print('Probability (pois. dist.): ', poissonProb)