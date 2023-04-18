#!/usr/bin/python3

from functions import *
from distributions import *

print('#### PROGRAMMING STATISTIC ####\n')
print('0 ==> Permutation \n1 ==> Combination \n2 ==> PROBABILITY DISTRIBUTION \n3 ==> BINOMIAL DISTRIBUTION \n4 ==> POISSON DISTRIBUTION \n5 ==> NORMAL DISTRIBUTION')
print('6 ==> STANDARD NORMAL DISTRIBUTION \n7 ==> CONFIDENCE INTERVALS \n8 ==> TEST STATISTICS (ONE POPULATION) \n8 ==> TEST STATISTICS (TWO POPULATION)\n10 ==> LINEAR CORRELATION/REGRESSION')

opt = int(input('Please select an option: '))

match opt:
    case 0:
        permutation()
        exit() # Add loop to allow user to pick a different option or exit
    case 1:
        print('Not implemented')
    case 2:
        probability_dist()
        exit() # Add loop to allow user to pick a different option or exit
    case 3:
        print('Not implemented')
    case 4:
        print('Not implemented')
    case 5:
        print('Not implemented')
    case 6:
        print('Not implemented')
    case 7:
        print('Not implemented')
    case 8:
        hypothesis_testing()
        exit() # Add loop to allow user to pick a different option or exit
    case 9:
        print('Not implemented')
    case 10:
        linear_corr_regr()
        exit() # Add loop to allow user to pick a different option or exit
    case _:
        print("invalid option")
        exit()

