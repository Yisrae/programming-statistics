#!/usr/bin/python3

from math import factorial, sqrt
from statistics import *
from itertools import permutations, combinations, combinations_with_replacement
from scipy import stats
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

# Return squared value of each list element
def _square_list(list):
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

x = [0,1,2,3] # Random variables
x_probability = [0.46,0.41,0.09,0.04] # Random variables probabilities
x_multiply_x_probability = [a * b for a, b in zip(x, x_probability)]

d = {
     'x': x,
     'P(x)': x_probability,
     'xP(x)': x_multiply_x_probability
 }

df = pd.DataFrame(data=d)
total_p_x = np.sum(x_probability) # Probability
total_x_p_x = np.sum(x_multiply_x_probability) # Mean
x_squared = _square_list(x)
x_squared_multiply_x_probability = [a * b for a, b in zip(x_squared, x_probability)]
x_squared_multiply_x_probability_total = np.sum(x_squared_multiply_x_probability)
variance = x_squared_multiply_x_probability_total - (total_x_p_x ** 2)
standard_deviation = sqrt(variance)

print(df)
print('\nTotal P(x): ', total_p_x)
print('Total xP(x) / Mean (μ): ', total_x_p_x)
print('Standard Deviation (prob. dist.): ', standard_deviation)


# Only works for exact values
print('\nBINOMIAL DISTRIBUTION')

n = 6 # Denotes the fixed number of trials
x = 3 # Denotes a specific number of successes in n trials
p = 0.3 # Denotes the probability of success in one of the n trials
q = 1 - p # Denotes the probability of failure in one of the n trials

binomial_probability = nCr(n, x) * (p ** x) * (q ** (n - x))
μ = n * p
variance = μ * q
binomial_standard_deviation = sqrt(variance)

print('Probability (bin. dist.): ', binomial_probability)
print('Mean (μ) (bin. dist.): ', μ)
print('Variance (bin. dist.): ', variance)
print('Standard Deviation (bin. dist.): ', binomial_standard_deviation)


# Only works for exact values
print('\nPOISSON DISTRIBUTION')
n = 14000 # Sample size
p = 0.00003 # Probability
μ = n * p # Mean
x = 1 # Number of occurrences of an event over some interval
E = 2.71828 # Approximately

# When approximating the Binomial distribution
if (n >= 100) == False:
    exit('Fail n > 100 check')

if (μ <= 10) == False:
    exit('Fail mean < 10 check')

poisson_probability = ((μ ** x) * (E ** -μ)) / factorial(x)
poisson_standard_deviation = sqrt(μ)

print('Probability (pois. dist.): ', poisson_probability)
print('Mean (μ) (pois. dist.): ', μ)
print('Standard Deviation (pois. dist.): ', poisson_standard_deviation)


print('\nNORMAL DISTRIBUTION')
# NOTE:
# Normal distribution highlights data values

print('\nSTANDARD NORMAL DISTRIBUTION')
# NOTE:
# Standard normal distribution highlights z scores

# Where mean = 0, standard_deviation = 1 and shaded_area = x

# Usage
# Returns area to the left
x = 1.13
print(f'Probability P(z < {x}) (norm. dist.):', stats.norm.cdf(x))

# Probability that z lies between 0 and x, where x is a point on the graph
def z_between_0_and_x(x):
    # 0 = Mean
    print(f'P({0} < z < {x})')
    print(f'P({0} < z < {x}) - P(z < {0})')

    return stats.norm.cdf(x) - stats.norm.cdf(0)

# Probability that z lies between 0 and x, where x is a point on the graph
def z_between_x_and_y(x, y):
    print(f'P({x} < z < {y})')
    return stats.norm.cdf(x) - stats.norm.cdf(y)

# Usage
x = 3.01
print(f'Probability that z lies between 0 and {x}:', z_between_0_and_x(x))

# Usage
x = -1.56
y = 4.56
print(f'Probability that z lies between {x} and {y}:', z_between_x_and_y(x,y))

def z_geater_than_x(x):
    return 1 - stats.norm.cdf(x)

# Usage
x = 0.59
print(f'z_geater_than_x:', z_geater_than_x(x))

# Finding percentage of data more than x standard deviations
# below the mean or more than x standard deviations above the mean
# x = standard deviations below mean
# y = standard deviations above mean
def percentage_of_data(x, y):
    print(f'1 - P(z < {x}) + P(z < {-y})')
    return (1 - stats.norm.cdf(x)) + stats.norm.cdf(-y)

x = 2
y = 3
print('Percentage of data:', percentage_of_data(x, y))

print('\nCONFIDENCE INTERVALS')
# Where one population

p̂ = None
E = None
q̂ = None
z = 0.005 # Z alpha/2

def indicated_value(z):
    area_to_the_left = 1 - z 
    return stats.norm.ppf(area_to_the_left)

# Usage
print('Indicated value:', indicated_value(z))
